from typing import Union

import torch
import torch_scatter
import numpy as np
from torch_spsr.core.hashtree import HashTree, VoxelStatus
from torch_spsr.bases.abc import BaseBasis
from torch_spsr.core.ops import screened_multiplication, marching_cubes_op, marching_cubes, torch_unique
from torch_spsr.ext import CuckooHashTable
from torch_spsr.core.solver import solve_sparse


class ScreeningData:
    """
    Sparse matrix representation (Num-pts x Num-vx)
    """
    def __init__(self, pts_ids, vx_ids, values, nb_sizes):
        self.pts_ids = pts_ids
        self.vx_ids = vx_ids
        self.values = values
        self.nb_sizes = nb_sizes


class RBF_FD:
    def __init__(self, hash_tree: HashTree, basis: BaseBasis, feat: dict = None):
        """
        Screened Poisson reconstructor class.
        :param hash_tree: The tree containing the octree structure as well as input points
        :param basis: basis function to use
        :param feat: dict that maps from integer depth to torch features, used only when basis needs features.
        """
        self.hash_tree = hash_tree
        self.fixed_level_set = None
        self.branch = hash_tree.DECODER

        self.basis = basis
        self.solutions = {}
        self.sample_weight = None

        # Initialize basis feature if not provided.
        if feat is None:
            feat = {}
            for d in range(hash_tree.depth):
                feat[d] = torch.zeros(
                    (hash_tree.get_coords_size(self.branch, d),
                     basis.get_feature_size()), device=hash_tree.device)
                basis.initialize_feature_value(feat[d])
        self.grid_features = feat


    @classmethod
    def _evaluate_screening_term(cls, data_a: ScreeningData, data_b: ScreeningData,
                                 domain_a, domain_b, pts_weight=None):
        if domain_a.size(0) == 0 or domain_b.size(0) == 0:
            return torch.zeros((0, ), device=domain_a.device)

        if pts_weight is None:
            pts_weight = torch.ones((data_a.nb_sizes.size(0), ), device=domain_a.device)
        elif isinstance(pts_weight, float):
            pts_weight = torch.full((data_a.nb_sizes.size(0), ), fill_value=pts_weight, dtype=torch.float32,
                                    device=domain_a.device)

        domain_table = CuckooHashTable(torch.stack([domain_a, domain_b], dim=1), enlarged=True)
        term_res = screened_multiplication(domain_table.object,
                                           data_a.values, data_b.values,
                                           data_a.vx_ids, data_b.vx_ids,
                                           data_a.nb_sizes, data_b.nb_sizes, pts_weight)

        return term_res

    def solve_multigrid(self, start_depth, end_depth, normal_data: dict,
                        screen_alpha: Union[float, torch.Tensor] = 0.0, screen_xyz: torch.Tensor = None,
                        screen_delta: Union[float, torch.Tensor] = 0.1,
                        solver: str = "pcg", verbose: bool = True):
        """
            Build and solve the linear system L alpha = d, using our coarse-to-fine solver.
        Note that the full V-cycle is not supported in this repo. Normal is however not smoothed
        because empirically we've found no difference.
            The energy function defined in our paper is solver within a truncated domain, with
        explicit dirichlet constraints that the boundary evaluates to 0. We choose not to eliminate
        such constraints because that will introduce many heterogeneous integral computations on
        the boundary.
        :param start_depth: int, the coarsest level for the solver
        :param end_depth: int, the finest level for the solver
        :param normal_data: dictionary that maps from depth to splatted normal data (x, 3)
        :param screen_alpha: float or Tensor. Weight of the screening term
        :param screen_xyz: None or Tensor. positional constraints to the system.
        :param screen_delta: float or Tensor. Target scalar value as described in the paper.
        :param solver: you can choose from 'cholmod' or 'pcg' or 'mixed'.
        :param verbose: Output debug information during solve.
        """
        if isinstance(screen_alpha, torch.Tensor):
            assert screen_xyz is not None, "Must provide points to be screened."
            assert screen_alpha.size(0) == screen_xyz.size(0)
            should_screen = True
        else:
            should_screen = screen_alpha > 0.0

        self.solutions = {}
        neighbour_range = 2 * self.basis.get_domain_range() - 1

        # Basis pre-evaluation for screening term.
        screen_data = {}
        if should_screen:
            base_coords = screen_xyz / self.hash_tree.voxel_size - 0.5
            for d in range(end_depth, start_depth + 1):
                pts_ids, vx_ids, tgt_offsets, nb_sizes = self.hash_tree.get_neighbours_data(
                    base_coords, 1, d, self.hash_tree.get_range_kernel(self.basis.get_domain_range()),
                    self.branch, transposed=True)
                query_coords = -tgt_offsets / self.hash_tree.get_stride(self.branch, d)
                query_val = self.basis.evaluate(feat=self.grid_features[d], xyz=query_coords, feat_ids=vx_ids)
                screen_data[d] = ScreeningData(pts_ids, vx_ids, query_val, nb_sizes)

        for d in range(start_depth, end_depth - 1, -1):
            screen_factor = (1 / 4.) ** d

            # Build RHS:
            rhs_val = 0
            for data_depth, depth_normal_data in normal_data.items():
                normal_ids, tree_ids, normal_offset, _ = self.hash_tree.get_neighbours(
                    data_depth, d,
                    self.basis.get_domain_range(),
                    self.branch)
                partial_sums = self.basis.integrate_const_deriv_product(
                    data=-depth_normal_data[normal_ids],
                    target_feat=self.grid_features[d],
                    rel_pos=normal_offset,
                    data_stride=self.hash_tree.get_stride(self.branch, data_depth),
                    target_stride=self.hash_tree.get_stride(self.branch, d),
                    target_ids=tree_ids
                )
                rhs_val += torch_scatter.scatter_add(
                    partial_sums, tree_ids, dim=0, dim_size=self.hash_tree.get_coords_size(self.branch, d))

            if should_screen:
                if isinstance(screen_alpha, torch.Tensor) or isinstance(screen_delta, torch.Tensor):
                    mult = (screen_delta * screen_alpha)[screen_data[d].pts_ids]
                else:
                    mult = screen_alpha * screen_delta
                rhs_val += screen_factor * torch_scatter.scatter_sum(
                    screen_data[d].values * mult,
                    screen_data[d].vx_ids, dim_size=self.hash_tree.get_coords_size(self.branch, d)
                )

            # Correction:
            for dd in range(start_depth, d, -1):
                src_ids, tgt_ids, rel_pos, _ = self.hash_tree.get_neighbours(
                    d, dd, target_range=neighbour_range, branch=self.branch)
                a_d_dd_val = self.basis.integrate_deriv_deriv_product(
                    source_feat=self.grid_features[d],
                    target_feat=self.grid_features[dd],
                    rel_pos=rel_pos,
                    source_stride=self.hash_tree.get_stride(self.branch, d),
                    target_stride=self.hash_tree.get_stride(self.branch, dd),
                    source_ids=src_ids, target_ids=tgt_ids)
                if should_screen:
                    a_d_dd_val = a_d_dd_val + screen_factor * self._evaluate_screening_term(
                        screen_data[d], screen_data[dd], src_ids, tgt_ids, screen_alpha)
                rhs_val -= torch_scatter.scatter_sum(self.solutions[dd][tgt_ids] * a_d_dd_val,
                                                     src_ids, dim_size=rhs_val.size(0))

            # Build LHS:
            src_ids, tgt_ids, rel_pos, _ = self.hash_tree.get_neighbours(d, d, target_range=neighbour_range,
                                                                         branch=self.branch)
            lhs_val = self.basis.integrate_deriv_deriv_product(
                source_feat=self.grid_features[d],
                target_feat=self.grid_features[d],
                rel_pos=rel_pos,
                source_stride=self.hash_tree.get_stride(self.branch, d),
                target_stride=self.hash_tree.get_stride(self.branch, d),
                source_ids=src_ids, target_ids=tgt_ids)

            if should_screen:
                lhs_val = lhs_val + screen_factor * self._evaluate_screening_term(
                    screen_data[d], screen_data[d], src_ids, tgt_ids, screen_alpha)

            if solver == "mixed":
                cur_solver = "mixed" if d == end_depth else "cholmod"
            else:
                cur_solver = solver
            self.solutions[d] = solve_sparse(src_ids, tgt_ids, lhs_val, rhs_val, cur_solver)

            # Dump residual for comparison
            if verbose:
                residual = torch_scatter.scatter_sum(self.solutions[d][tgt_ids] * lhs_val, src_ids,
                                                     dim_size=rhs_val.size(0)) - rhs_val
                print(f"Solving complete at level {d}, residual = {torch.linalg.norm(residual).item()}.")