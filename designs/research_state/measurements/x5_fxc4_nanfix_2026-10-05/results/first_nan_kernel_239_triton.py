triton_red_fused__to_copy_add_bitwise_or_clamp_div_eq_gather_ge_gt_index_le_logical_and_logsumexp_mul_neg_reciprocal_rsub_sub_sum_where_239 = async_compile.triton('triton_red_fused__to_copy_add_bitwise_or_clamp_div_eq_gather_ge_gt_index_le_logical_and_logsumexp_mul_neg_reciprocal_rsub_sub_sum_where_239', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*i1', 'in_ptr40': '*fp32', 'in_ptr41': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]], (31,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (33,): [['tt.divisibility', 16]], (34,): [['tt.divisibility', 16]], (35,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (37,): [['tt.divisibility', 16]], (38,): [['tt.divisibility', 16]], (39,): [['tt.divisibility', 16]], (40,): [['tt.divisibility', 16]], (41,): [['tt.divisibility', 16]], (42,): [['tt.divisibility', 16]], (43,): [['tt.divisibility', 16]], (44,): [['tt.divisibility', 16]], (45,): [['tt.divisibility', 16]], (46,): [['tt.divisibility', 16]], (47,): [['tt.divisibility', 16]], (48,): [['tt.divisibility', 16]], (49,): [['tt.divisibility', 16]], (50,): [['tt.divisibility', 16]], (51,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (53,): [['tt.divisibility', 16]], (54,): [['tt.divisibility', 16]], (55,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_bitwise_or_clamp_div_eq_gather_ge_gt_index_le_logical_and_logsumexp_mul_neg_reciprocal_rsub_sub_sum_where_239', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 63, 'num_reduction': 10, 'backend_hash': '412B4371F12BB7A177EE885F3E6F881ACBCE6D15164D752B7396BA7DAAFE73C2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_bitwise_or_clamp_div_eq_gather_ge_gt_index_le_logical_and_logsumexp_mul_neg_reciprocal_rsub_sub_sum_where_239(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 400
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 6)
    x1 = xindex // 6
    tmp6 = tl.load(in_ptr1 + (x0 + 12*x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + 12*x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0 + 12*x1), None, eviction_policy='evict_last')
    x3 = xindex
    tmp53 = tl.load(in_ptr12 + (2761*x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr14 + (2761*x1), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr15 + (328*x0 + 3936*x1), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr16 + (x3), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr21 + (x1), None, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr25 + (x3), None, eviction_policy='evict_last')
    _tmp160 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp162 = tl.load(in_ptr26 + (x3), None, eviction_policy='evict_last')
    _tmp167 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp169 = tl.load(in_ptr27 + (x3), None, eviction_policy='evict_last')
    _tmp174 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp176 = tl.load(in_ptr28 + (x3), None, eviction_policy='evict_last')
    _tmp181 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp183 = tl.load(in_ptr29 + (x3), None, eviction_policy='evict_last')
    _tmp188 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp190 = tl.load(in_ptr30 + (x3), None, eviction_policy='evict_last')
    _tmp195 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp197 = tl.load(in_ptr31 + (x3), None, eviction_policy='evict_last')
    _tmp202 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp204 = tl.load(in_ptr32 + (x3), None, eviction_policy='evict_last')
    _tmp209 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp211 = tl.load(in_ptr33 + (x3), None, eviction_policy='evict_last')
    _tmp216 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp218 = tl.load(in_ptr34 + (x3), None, eviction_policy='evict_last')
    _tmp223 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr6 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr7 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr10 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr11 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp54 = tl.load(in_ptr13 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp79 = tl.load(in_ptr17 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp103 = tl.load(in_ptr23 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp131 = tl.load(in_ptr24 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 19, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 19)) | ~(r0_mask), "index out of bounds: 0 <= tmp4 < 19")
        tmp7 = tmp6 + tmp1
        tmp8 = tmp6 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp6)
        tl.device_assert((0 <= tmp9) & (tmp9 < 19), "index out of bounds: 0 <= tmp9 < 19")
        tmp11 = tl.load(in_ptr2 + (tmp4 + 19*tmp9), r0_mask, eviction_policy='evict_last')
        tmp13 = tmp12 + tmp1
        tmp14 = tmp12 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp12)
        tl.device_assert((0 <= tmp15) & (tmp15 < 19), "index out of bounds: 0 <= tmp15 < 19")
        tmp17 = tl.load(in_ptr2 + (tmp4 + 19*tmp15), r0_mask, eviction_policy='evict_last')
        tmp18 = tmp11 * tmp17
        tmp20 = tl.full([XBLOCK, R0_BLOCK], 100, tl.int32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp19 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp19)
        tl.device_assert((0 <= tmp23) & (tmp23 < 100), "index out of bounds: 0 <= tmp23 < 100")
        tmp25 = tl.load(in_ptr5 + (tmp4 + 19*tmp23), r0_mask, eviction_policy='evict_last')
        tmp26 = tmp18 * tmp25
        tmp28 = 0.84
        tmp29 = tmp27 * tmp28
        tmp31 = tl.full([XBLOCK, R0_BLOCK], 3, tl.int32)
        tmp32 = tmp30 + tmp31
        tmp33 = tmp30 < 0
        tmp34 = tl.where(tmp33, tmp32, tmp30)
        tl.device_assert(((0 <= tmp34) & (tmp34 < 3)) | ~(r0_mask), "index out of bounds: 0 <= tmp34 < 3")
        tmp36 = tl.load(in_ptr8 + (tmp34 + 3*x1), r0_mask, eviction_policy='evict_last')
        tmp37 = tmp29 * tmp36
        tmp38 = tl.load(in_ptr9 + (tmp34 + 3*x3), r0_mask, eviction_policy='evict_last')
        tmp39 = 1e-06
        tmp40 = tmp38 + tmp39
        tmp41 = tl.full([1, 1], 1, tl.int32)
        tmp42 = (tmp41 / tmp40)
        tmp43 = 1.0
        tmp44 = tmp42 * tmp43
        tmp45 = tmp37 * tmp44
        tmp47 = tmp45 + tmp46
        tmp49 = 0.925
        tmp50 = tmp48 * tmp49
        tmp51 = tmp26 * tmp50
        tmp52 = tmp47 * tmp51
        tmp55 = tmp53 * tmp54
        tmp57 = tmp43 - tmp54
        tmp58 = tmp56 * tmp57
        tmp59 = tmp55 + tmp58
        tmp60 = 0.5
        tmp61 = tmp59 * tmp60
        tmp62 = tmp43 - tmp61
        tmp63 = tmp52 * tmp62
        tmp66 = tmp64 * tmp65
        tmp67 = tmp63 - tmp66
        tmp68 = 0.15
        tmp69 = tmp63 * tmp68
        tmp70 = tmp69 + tmp39
        tmp71 = (tmp67 / tmp70)
        tmp72 = tmp65 + tmp39
        tmp73 = (tmp41 / tmp72)
        tmp74 = tmp73 * tmp43
        tmp75 = tmp63 * tmp74
        tmp76 = 0.85
        tmp77 = tmp63 * tmp76
        tmp78 = tmp77 * tmp74
        tmp80 = tl.full([XBLOCK, R0_BLOCK], 400, tl.int32)
        tmp81 = tmp79 + tmp80
        tmp82 = tmp79 < 0
        tmp83 = tl.where(tmp82, tmp81, tmp79)
        tl.device_assert(((0 <= tmp83) & (tmp83 < 400)) | ~(r0_mask), "index out of bounds: 0 <= tmp83 < 400")
        tmp85 = tl.load(in_ptr18 + (tmp83), r0_mask, eviction_policy='evict_last')
        tmp86 = tl.load(in_ptr19 + (tmp83), r0_mask, eviction_policy='evict_last')
        tmp87 = tmp86 * tmp66
        tmp88 = tmp85 + tmp87
        tmp89 = tl.load(in_ptr20 + (tmp83), r0_mask, eviction_policy='evict_last')
        tmp91 = tmp66 - tmp90
        tmp92 = 0.0
        tmp93 = triton_helpers.maximum(tmp91, tmp92)
        tmp94 = tmp89 * tmp93
        tmp95 = tmp88 + tmp94
        tmp96 = (tmp95 / tmp72)
        tmp97 = tl.load(in_ptr22 + (tmp34 + 3*x1), r0_mask, eviction_policy='evict_last')
        tmp98 = tmp29 * tmp97
        tmp99 = tmp98 * tmp44
        tmp100 = tmp99 + tmp46
        tmp101 = tmp100 * tmp51
        tmp102 = tmp101 * tmp62
        tmp104 = tmp85 + tmp86
        tmp105 = tmp104 + tmp89
        tmp106 = tmp105 > tmp92
        tmp107 = 1.5
        tmp108 = triton_helpers.minimum(tmp96, tmp107)
        tmp109 = tmp26 > tmp92
        tmp110 = tmp109.to(tl.float32)
        tmp111 = tmp108 * tmp110
        tmp112 = tmp102 * tmp74
        tmp113 = triton_helpers.minimum(tmp112, tmp107)
        tmp114 = tl.where(tmp106, tmp111, tmp113)
        tmp115 = tmp103 * tmp114
        tmp116 = 3.0
        tmp117 = triton_helpers.minimum(tmp96, tmp116)
        tmp118 = tmp117 * tmp110
        tmp119 = 2.0
        tmp120 = tmp52 * tmp119
        tmp121 = tmp120 * tmp74
        tmp122 = triton_helpers.minimum(tmp121, tmp116)
        tmp123 = tl.where(tmp106, tmp118, tmp122)
        tmp124 = tmp103 * tmp123
        tmp125 = triton_helpers.minimum(tmp75, tmp107)
        tmp126 = tl.where(tmp106, tmp111, tmp125)
        tmp127 = tmp103 * tmp126
        tmp128 = triton_helpers.minimum(tmp78, tmp107)
        tmp129 = tl.where(tmp106, tmp111, tmp128)
        tmp130 = tmp103 * tmp129
        tmp132 = tmp86 >= tmp43
        tmp133 = tmp85 >= tmp66
        tmp134 = tmp132 | tmp133
        tmp135 = tmp134.to(tl.float32)
        tmp136 = tmp131 * tmp135
        tmp137 = tmp66 > tmp92
        tmp138 = tmp137.to(tl.float32)
        tmp139 = tmp136 * tmp138
        tmp140 = tmp139 * tmp110
        tmp141 = tmp102 - tmp66
        tmp142 = tmp102 * tmp68
        tmp143 = tmp142 + tmp39
        tmp144 = (tmp141 / tmp143)
        tmp145 = triton_helpers.maximum(tmp144, tmp92)
        tmp146 = triton_helpers.minimum(tmp145, tmp43)
        tmp147 = tmp131 * tmp146
        tmp148 = tl.where(tmp106, tmp140, tmp147)
        tmp149 = triton_helpers.maximum(tmp71, tmp92)
        tmp150 = triton_helpers.minimum(tmp149, tmp43)
        tmp151 = tmp131 * tmp150
        tmp152 = tl.where(tmp106, tmp140, tmp151)
        tmp153 = tmp103 * tmp152
        tmp155 = tmp103 * tmp148
        tmp156 = tmp155 * tmp54
        tmp157 = tmp154 == tmp156
        tmp158 = tmp157.to(tl.int64)
        tmp159 = tl.broadcast_to(tmp158, [XBLOCK, R0_BLOCK])
        tmp161 = _tmp160 + tmp159
        _tmp160 = tl.where(r0_mask, tmp161, _tmp160)
        tmp163 = tmp115 * tmp54
        tmp164 = tmp162 == tmp163
        tmp165 = tmp164.to(tl.int64)
        tmp166 = tl.broadcast_to(tmp165, [XBLOCK, R0_BLOCK])
        tmp168 = _tmp167 + tmp166
        _tmp167 = tl.where(r0_mask, tmp168, _tmp167)
        tmp170 = tmp153 * tmp57
        tmp171 = tmp169 == tmp170
        tmp172 = tmp171.to(tl.int64)
        tmp173 = tl.broadcast_to(tmp172, [XBLOCK, R0_BLOCK])
        tmp175 = _tmp174 + tmp173
        _tmp174 = tl.where(r0_mask, tmp175, _tmp174)
        tmp177 = tmp153 * tmp54
        tmp178 = tmp176 == tmp177
        tmp179 = tmp178.to(tl.int64)
        tmp180 = tl.broadcast_to(tmp179, [XBLOCK, R0_BLOCK])
        tmp182 = _tmp181 + tmp180
        _tmp181 = tl.where(r0_mask, tmp182, _tmp181)
        tmp184 = tmp124 * tmp57
        tmp185 = tmp183 == tmp184
        tmp186 = tmp185.to(tl.int64)
        tmp187 = tl.broadcast_to(tmp186, [XBLOCK, R0_BLOCK])
        tmp189 = _tmp188 + tmp187
        _tmp188 = tl.where(r0_mask, tmp189, _tmp188)
        tmp191 = tmp124 * tmp54
        tmp192 = tmp190 == tmp191
        tmp193 = tmp192.to(tl.int64)
        tmp194 = tl.broadcast_to(tmp193, [XBLOCK, R0_BLOCK])
        tmp196 = _tmp195 + tmp194
        _tmp195 = tl.where(r0_mask, tmp196, _tmp195)
        tmp198 = tmp127 * tmp57
        tmp199 = tmp197 == tmp198
        tmp200 = tmp199.to(tl.int64)
        tmp201 = tl.broadcast_to(tmp200, [XBLOCK, R0_BLOCK])
        tmp203 = _tmp202 + tmp201
        _tmp202 = tl.where(r0_mask, tmp203, _tmp202)
        tmp205 = tmp127 * tmp54
        tmp206 = tmp204 == tmp205
        tmp207 = tmp206.to(tl.int64)
        tmp208 = tl.broadcast_to(tmp207, [XBLOCK, R0_BLOCK])
        tmp210 = _tmp209 + tmp208
        _tmp209 = tl.where(r0_mask, tmp210, _tmp209)
        tmp212 = tmp130 * tmp57
        tmp213 = tmp211 == tmp212
        tmp214 = tmp213.to(tl.int64)
        tmp215 = tl.broadcast_to(tmp214, [XBLOCK, R0_BLOCK])
        tmp217 = _tmp216 + tmp215
        _tmp216 = tl.where(r0_mask, tmp217, _tmp216)
        tmp219 = tmp130 * tmp54
        tmp220 = tmp218 == tmp219
        tmp221 = tmp220.to(tl.int64)
        tmp222 = tl.broadcast_to(tmp221, [XBLOCK, R0_BLOCK])
        tmp224 = _tmp223 + tmp222
        _tmp223 = tl.where(r0_mask, tmp224, _tmp223)
        tl.store(out_ptr0 + (r0_2 + 400*x3), tmp26, r0_mask)
        tl.store(in_out_ptr4 + (r0_2 + 400*x3), tmp52, r0_mask)
        tl.store(out_ptr1 + (r0_2 + 400*x3), tmp71, r0_mask)
        tl.store(out_ptr2 + (r0_2 + 400*x3), tmp75, r0_mask)
        tl.store(out_ptr3 + (r0_2 + 400*x3), tmp78, r0_mask)
        tl.store(out_ptr5 + (r0_2 + 400*x3), tmp102, r0_mask)
        tl.store(in_out_ptr6 + (r0_2 + 400*x3), tmp115, r0_mask)
        tl.store(in_out_ptr2 + (r0_2 + 400*x3), tmp124, r0_mask)
        tl.store(in_out_ptr0 + (r0_2 + 400*x3), tmp127, r0_mask)
        tl.store(in_out_ptr3 + (r0_2 + 400*x3), tmp130, r0_mask)
        tl.store(in_out_ptr5 + (r0_2 + 400*x3), tmp148, r0_mask)
        tl.store(in_out_ptr1 + (r0_2 + 400*x3), tmp153, r0_mask)
    tmp160 = tl.sum(_tmp160, 1)[:, None]
    tmp167 = tl.sum(_tmp167, 1)[:, None]
    tmp174 = tl.sum(_tmp174, 1)[:, None]
    tmp181 = tl.sum(_tmp181, 1)[:, None]
    tmp188 = tl.sum(_tmp188, 1)[:, None]
    tmp195 = tl.sum(_tmp195, 1)[:, None]
    tmp202 = tl.sum(_tmp202, 1)[:, None]
    tmp209 = tl.sum(_tmp209, 1)[:, None]
    tmp216 = tl.sum(_tmp216, 1)[:, None]
    tmp223 = tl.sum(_tmp223, 1)[:, None]
    tmp225 = tl.load(in_ptr35 + (78 + x0 + 138*x1), None, eviction_policy='evict_last')
    tmp237 = tl.load(in_ptr38 + (78 + x0), None, eviction_policy='evict_last')
    tmp241 = tl.load(in_ptr39 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp247 = tl.load(in_ptr35 + (72 + x0 + 138*x1), None, eviction_policy='evict_last')
    tmp257 = tl.load(in_ptr38 + (72 + x0), None, eviction_policy='evict_last')
    tmp226 = 78 + x0
    tmp227 = tl.full([1, 1], 72, tl.int64)
    tmp228 = tmp226 < tmp227
    tmp229 = tl.load(in_ptr36 + (72*x1 + (((78 + x0) % 72))), tmp228, eviction_policy='evict_last', other=0.0)
    tmp230 = tl.load(in_ptr37 + (72*x1 + (((78 + x0) % 72))), tmp228, eviction_policy='evict_last', other=0.0)
    tmp231 = tmp229 + tmp230
    tmp232 = tl.full(tmp231.shape, 0.0, tmp231.dtype)
    tmp233 = tl.where(tmp228, tmp231, tmp232)
    tmp234 = 0.0
    tmp235 = tl.where(tmp228, tmp233, tmp234)
    tmp236 = tmp225 + tmp235
    tmp238 = tmp236 * tmp237
    tmp239 = tmp64 > tmp234
    tmp240 = tmp239.to(tl.float32)
    tmp242 = tmp241.to(tl.float32)
    tmp243 = tmp240 * tmp242
    tmp244 = tmp238 * tmp243
    tmp245 = tmp160.to(tl.float32)
    tmp246 = (tmp244 / tmp245)
    tmp248 = 72 + x0
    tmp249 = tmp248 < tmp227
    tmp250 = tl.load(in_ptr36 + (x0 + 72*x1), tmp249, eviction_policy='evict_last', other=0.0)
    tmp251 = tl.load(in_ptr37 + (x0 + 72*x1), tmp249, eviction_policy='evict_last', other=0.0)
    tmp252 = tmp250 + tmp251
    tmp253 = tl.full(tmp252.shape, 0.0, tmp252.dtype)
    tmp254 = tl.where(tmp249, tmp252, tmp253)
    tmp255 = tl.where(tmp249, tmp254, tmp234)
    tmp256 = tmp247 + tmp255
    tmp258 = tmp256 * tmp257
    tmp259 = tmp258 * tmp243
    tmp260 = tmp167.to(tl.float32)
    tmp261 = (tmp259 / tmp260)
    tmp262 = tl.load(in_ptr40 + (6 + 12*x3), None, eviction_policy='evict_last')
    tmp274 = tl.load(in_ptr40 + (1 + 12*x3), None, eviction_policy='evict_last')
    tmp285 = tl.load(in_ptr40 + (8 + 12*x3), None, eviction_policy='evict_last')
    tmp294 = tl.load(in_ptr40 + (3 + 12*x3), None, eviction_policy='evict_last')
    tmp304 = tl.load(in_ptr40 + (7 + 12*x3), None, eviction_policy='evict_last')
    tmp313 = tl.load(in_ptr40 + (2 + 12*x3), None, eviction_policy='evict_last')
    tmp323 = tl.load(in_ptr40 + (5 + 12*x3), None, eviction_policy='evict_last')
    tmp332 = tl.load(in_ptr40 + (12*x3), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp265 = tl.load(in_out_ptr0 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp266 = tl.load(in_ptr13 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp283 = tl.load(in_ptr23 + (r0_2 + 400*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp288 = tl.load(in_out_ptr1 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp307 = tl.load(in_out_ptr2 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp326 = tl.load(in_out_ptr3 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp342 = tl.load(out_ptr1 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp346 = tl.load(in_ptr17 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp358 = tl.load(in_ptr41 + (3 + 6*r0_2 + 2400*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp361 = tl.load(in_ptr24 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp365 = tl.load(in_out_ptr4 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp382 = tl.load(out_ptr3 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp385 = tl.load(in_ptr41 + (6*r0_2 + 2400*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp397 = tl.load(out_ptr2 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp399 = tl.load(in_ptr41 + (1 + 6*r0_2 + 2400*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp410 = tl.load(in_ptr41 + (2 + 6*r0_2 + 2400*x3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp418 = tl.load(in_out_ptr5 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp427 = tl.load(in_out_ptr6 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp435 = tl.load(out_ptr5 + (r0_2 + 400*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp263 = tmp202.to(tl.float32)
        tmp264 = (tmp262 / tmp263)
        tmp267 = 1.0
        tmp268 = tmp267 - tmp266
        tmp269 = tmp265 * tmp268
        tmp270 = tmp197 == tmp269
        tmp271 = tmp270.to(tl.float32)
        tmp272 = tmp264 * tmp271
        tmp273 = tmp272 * tmp268
        tmp275 = tmp209.to(tl.float32)
        tmp276 = (tmp274 / tmp275)
        tmp277 = tmp265 * tmp266
        tmp278 = tmp204 == tmp277
        tmp279 = tmp278.to(tl.float32)
        tmp280 = tmp276 * tmp279
        tmp281 = tmp280 * tmp266
        tmp282 = tmp273 + tmp281
        tmp284 = tmp282 * tmp283
        tmp286 = tmp174.to(tl.float32)
        tmp287 = (tmp285 / tmp286)
        tmp289 = tmp288 * tmp268
        tmp290 = tmp169 == tmp289
        tmp291 = tmp290.to(tl.float32)
        tmp292 = tmp287 * tmp291
        tmp293 = tmp292 * tmp268
        tmp295 = tmp181.to(tl.float32)
        tmp296 = (tmp294 / tmp295)
        tmp297 = tmp288 * tmp266
        tmp298 = tmp176 == tmp297
        tmp299 = tmp298.to(tl.float32)
        tmp300 = tmp296 * tmp299
        tmp301 = tmp300 * tmp266
        tmp302 = tmp293 + tmp301
        tmp303 = tmp302 * tmp283
        tmp305 = tmp188.to(tl.float32)
        tmp306 = (tmp304 / tmp305)
        tmp308 = tmp307 * tmp268
        tmp309 = tmp183 == tmp308
        tmp310 = tmp309.to(tl.float32)
        tmp311 = tmp306 * tmp310
        tmp312 = tmp311 * tmp268
        tmp314 = tmp195.to(tl.float32)
        tmp315 = (tmp313 / tmp314)
        tmp316 = tmp307 * tmp266
        tmp317 = tmp190 == tmp316
        tmp318 = tmp317.to(tl.float32)
        tmp319 = tmp315 * tmp318
        tmp320 = tmp319 * tmp266
        tmp321 = tmp312 + tmp320
        tmp322 = tmp321 * tmp283
        tmp324 = tmp216.to(tl.float32)
        tmp325 = (tmp323 / tmp324)
        tmp327 = tmp326 * tmp268
        tmp328 = tmp211 == tmp327
        tmp329 = tmp328.to(tl.float32)
        tmp330 = tmp325 * tmp329
        tmp331 = tmp330 * tmp268
        tmp333 = tmp223.to(tl.float32)
        tmp334 = (tmp332 / tmp333)
        tmp335 = tmp326 * tmp266
        tmp336 = tmp218 == tmp335
        tmp337 = tmp336.to(tl.float32)
        tmp338 = tmp334 * tmp337
        tmp339 = tmp338 * tmp266
        tmp340 = tmp331 + tmp339
        tmp341 = tmp340 * tmp283
        tmp343 = tmp342 >= tmp234
        tmp344 = tmp342 <= tmp267
        tmp345 = tmp343 & tmp344
        tmp347 = tl.full([XBLOCK, R0_BLOCK], 400, tl.int32)
        tmp348 = tmp346 + tmp347
        tmp349 = tmp346 < 0
        tmp350 = tl.where(tmp349, tmp348, tmp346)
        tl.device_assert(((0 <= tmp350) & (tmp350 < 400)) | ~(r0_mask), "index out of bounds: 0 <= tmp350 < 400")
        tmp352 = tl.load(in_ptr18 + (tmp350), r0_mask, eviction_policy='evict_last')
        tmp353 = tl.load(in_ptr19 + (tmp350), r0_mask, eviction_policy='evict_last')
        tmp354 = tmp352 + tmp353
        tmp355 = tl.load(in_ptr20 + (tmp350), r0_mask, eviction_policy='evict_last')
        tmp356 = tmp354 + tmp355
        tmp357 = tmp356 > tmp234
        tmp359 = tmp358 + tmp303
        tmp360 = tl.where(tmp357, tmp234, tmp359)
        tmp362 = tmp360 * tmp361
        tmp363 = tl.where(tmp345, tmp362, tmp234)
        tmp364 = -tmp363
        tmp366 = tmp53 * tmp266
        tmp367 = tmp56 * tmp268
        tmp368 = tmp366 + tmp367
        tmp369 = 0.5
        tmp370 = tmp368 * tmp369
        tmp371 = tmp267 - tmp370
        tmp372 = tmp365 * tmp371
        tmp373 = 0.15
        tmp374 = tmp372 * tmp373
        tmp375 = 1e-06
        tmp376 = tmp374 + tmp375
        tmp377 = (tmp342 / tmp376)
        tmp378 = tmp364 * tmp377
        tmp379 = tmp378 * tmp373
        tmp380 = (tmp363 / tmp376)
        tmp381 = tmp379 + tmp380
        tmp383 = 1.5
        tmp384 = tmp382 <= tmp383
        tmp386 = tmp385 + tmp341
        tmp387 = tl.where(tmp357, tmp234, tmp386)
        tmp388 = tl.where(tmp384, tmp387, tmp234)
        tmp389 = tmp65 + tmp375
        tmp390 = tl.full([1, 1], 1, tl.int32)
        tmp391 = (tmp390 / tmp389)
        tmp392 = tmp391 * tmp267
        tmp393 = tmp388 * tmp392
        tmp394 = 0.85
        tmp395 = tmp393 * tmp394
        tmp396 = tmp381 + tmp395
        tmp398 = tmp397 <= tmp383
        tmp400 = tmp399 + tmp284
        tmp401 = tl.where(tmp357, tmp234, tmp400)
        tmp402 = tl.where(tmp398, tmp401, tmp234)
        tmp403 = tmp402 * tmp392
        tmp404 = tmp396 + tmp403
        tmp405 = 2.0
        tmp406 = tmp365 * tmp405
        tmp407 = tmp406 * tmp392
        tmp408 = 3.0
        tmp409 = tmp407 <= tmp408
        tmp411 = tmp410 + tmp322
        tmp412 = tl.where(tmp357, tmp234, tmp411)
        tmp413 = tl.where(tmp409, tmp412, tmp234)
        tmp414 = tmp413 * tmp392
        tmp415 = tmp414 * tmp405
        tmp416 = tmp404 * tmp371
        tmp417 = tmp415 + tmp416
        tmp419 = tmp283 * tmp418
        tmp420 = tmp419 * tmp266
        tmp421 = tmp154 == tmp420
        tmp422 = tmp421.to(tl.float32)
        tmp423 = tmp246 * tmp422
        tmp424 = tmp423 * tmp266
        tmp425 = tmp424 * tmp283
        tmp426 = tl.where(tmp357, tmp234, tmp425)
        tmp428 = tmp427 * tmp266
        tmp429 = tmp162 == tmp428
        tmp430 = tmp429.to(tl.float32)
        tmp431 = tmp261 * tmp430
        tmp432 = tmp431 * tmp266
        tmp433 = tmp432 * tmp283
        tmp434 = tl.where(tmp357, tmp234, tmp433)
        tmp436 = tmp64 * tmp65
        tmp437 = tmp435 - tmp436
        tmp438 = tmp435 * tmp373
        tmp439 = tmp438 + tmp375
        tmp440 = (tmp437 / tmp439)
        tmp441 = tmp440 >= tmp234
        tmp442 = tmp440 <= tmp267
        tmp443 = tmp441 & tmp442
        tmp444 = tmp426 * tmp361
        tmp445 = tl.where(tmp443, tmp444, tmp234)
        tmp446 = -tmp445
        tmp447 = (tmp440 / tmp439)
        tmp448 = tmp446 * tmp447
        tmp449 = tmp448 * tmp373
        tmp450 = (tmp445 / tmp439)
        tmp451 = tmp449 + tmp450
        tmp452 = tmp435 * tmp392
        tmp453 = tmp452 <= tmp383
        tmp454 = tl.where(tmp453, tmp434, tmp234)
        tmp455 = tmp454 * tmp392
        tmp456 = tmp451 + tmp455
        tmp457 = tmp456 * tmp371
        tl.store(in_out_ptr4 + (r0_2 + 400*x3), tmp417, r0_mask)
        tl.store(in_out_ptr5 + (r0_2 + 400*x3), tmp457, r0_mask)
''', device_str='cuda')
