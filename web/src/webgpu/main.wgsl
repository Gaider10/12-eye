const workgroup_size: u32 = __WORKGROUP_SIZE__;

struct u64 {
    lo: u32,
    hi: u32,
}

fn u64_from_u32(a: u32) -> u64 {
    return u64(a, 0);
}

fn u64_from_i32(a: i32) -> u64 {
    return u64(u32(a), u32(a >> 31));
}

fn u64_add(a: u64, b: u64) -> u64 {
    let lo = a.lo + b.lo;
    let hi = a.hi + b.hi + u32(lo < a.lo);
    return u64(lo, hi);
}

fn u64_mul(a: u64, b: u64) -> u64 {
    let a_0 = a.lo & 0xFFFF;
    let a_1 = a.lo >> 16;
    let b_0 = b.lo & 0xFFFF;
    let b_1 = b.lo >> 16;

    let c_0_o = a_0 * b_0;
    let a_1_b_0 = a_1 * b_0;
    let a_0_b_1 = a_0 * b_1;
    let c_1_o = (a_1_b_0 & 0xFFFF) + (a_0_b_1 & 0xFFFF) + (c_0_o >> 16);
    // let c_lo = (c_0_o & 0xFFFF) + (c_1_o << 16);
    let c_lo = a.lo * b.lo;
    let c_hi = a.hi * b.lo + a.lo * b.hi + a_1 * b_1 + (a_1_b_0 >> 16) + (a_0_b_1 >> 16) + (c_1_o >> 16);
    return u64(c_lo, c_hi);
}

fn u64_or(a: u64, b: u64) -> u64 {
    return u64(a.lo | b.lo, a.hi | b.hi);
}

fn u64_xor(a: u64, b: u64) -> u64 {
    return u64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn u64_shr(a: u64, n: u32) -> u64 {
    if (n < 32) {
        return u64((a.lo >> n) | (a.hi << (32 - n)), a.hi >> n);
    } else {
        return u64(a.hi >> (n - 32), 0);
    }
}

fn u64_shl(a: u64, n: u32) -> u64 {
    if (n < 32) {
        return u64(a.lo << n, (a.hi << n) | (a.lo >> (32 - n)));
    } else {
        return u64(0, a.lo << (n - 32));
    }
}

fn u64_rol(a: u64, n: u32) -> u64 {
    // return u64_or(u64_shl(a, n), u64_shr(a, 64 - n));
    if (n < 32) {
        return u64((a.lo << n) | (a.hi >> (32 - n)), (a.hi << n) | (a.lo >> (32 - n)));
    } else {
        return u64((a.hi << (n - 32)) | (a.lo >> (64 - n)), (a.lo << (n - 32)) | (a.hi >> (64 - n)));
    }
}

struct u128 {
    lo: u64,
    hi: u64,
}

fn u128_xor(a: u128, b: u128) -> u128 {
    return u128(u64_xor(a.lo, b.lo), u64_xor(a.hi, b.hi));
}

const XRSR_MIX1 = u64(0x1ce4e5b9, 0xbf58476d);
const XRSR_MIX2 = u64(0x133111eb, 0x94d049bb);
const XRSR_MIX1_INVERSE = u64(0x3f119089, 0x96de1b17);
const XRSR_MIX2_INVERSE = u64(0xd24d8ec3, 0x319642b2);
const XRSR_SILVER_RATIO = u64(0xf3bcc909, 0x6a09e667);
const XRSR_GOLDEN_RATIO = u64(0x7f4a7c15, 0x9e3779b9);

fn mix64(s: u64) -> u64 {
    let s1 = u64_mul(u64_xor(s, u64_shr(s, 30)), XRSR_MIX1);
	let s2 = u64_mul(u64_xor(s1, u64_shr(s1, 27)), XRSR_MIX2);
	return u64_xor(s2, u64_shr(s2, 31));
}

fn xrsr_seed(seed: u64) -> u128 {
    let seed1 = u64_xor(seed, XRSR_SILVER_RATIO);
	let lo = mix64(seed1);
	let hi = mix64(u64_add(seed1, XRSR_GOLDEN_RATIO));
    return u128(lo, hi);
}

fn xrsr_next(xrsr: u128) -> u128 {
	let h = u64_xor(xrsr.hi, xrsr.lo);
	let lo = u64_xor(u64_xor(u64_rol(xrsr.lo, 49), h), u64_shl(h, 21));
	let hi = u64_rol(h, 28);
    return u128(lo, hi);
}

fn xrsr_long(xrsr: u128) -> u64 {
	return u64_add(u64_rol(u64_add(xrsr.lo, xrsr.hi), 17), xrsr.lo);
}

fn xrsr_next_bits(xrsr: ptr<function, u128>) -> u32 {
    let ret = xrsr_long(*xrsr).hi;
    *xrsr = xrsr_next(*xrsr);
    return ret;
}

fn xrsr_nextLong(xrsr: ptr<function, u128>) -> u64 {
    return u64_add(u64(0, xrsr_next_bits(xrsr)), u64_from_i32(i32(xrsr_next_bits(xrsr))));
}

fn xrsr_setFeatureSeed(world_seed: u64, x: i32, z: i32, salt: u32) -> u128 {
    var xrsr = xrsr_seed(world_seed);
    let a = u64_or(xrsr_nextLong(&xrsr), u64_from_u32(1));
    let b = u64_or(xrsr_nextLong(&xrsr), u64_from_u32(1));
    let decorationSeed = u64_xor(u64_add(u64_mul(a, u64_from_i32(x)), u64_mul(b, u64_from_i32(z))), world_seed);
    let featureSeed = u64_add(decorationSeed, u64_from_u32(salt));
    return xrsr_seed(featureSeed);
}

const precomp_size: u32 = __PRECOMP_SIZE__;

struct PrecompStorage {
    items: array<u128, precomp_size>,
}

struct Input {
    structure_seed: u64,
    packed_chunk_x: i32, // (portal_chunk_x << 16) | start_chunk_x
    packed_chunk_z: i32, // (portal_chunk_z << 16) | start_chunk_z
}

struct InputStorage {
    count: u32,
    items: array<Input>,
}

struct Result {
    seed: u64,
    packed_chunk_x: i32,
    packed_chunk_z: i32,
}

struct ResultStorage {
    count: atomic<u32>,
    items: array<Result>,
}

@group(0) @binding(0) var<storage, read> precomp: PrecompStorage;
@group(1) @binding(0) var<storage, read> inputs: InputStorage;
@group(1) @binding(1) var<storage, read_write> results: ResultStorage;

struct WorkgroupData {
    items: array<u128, precomp_size>,
}

var<workgroup> workgroup_data: WorkgroupData;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(num_workgroups)
    num_workgroups : vec3u,

    @builtin(workgroup_id)
    workgroup_id : vec3u,

    @builtin(local_invocation_index)
    local_invocation_index : u32,
) {
    for (var i: u32 = 0; i < precomp_size / workgroup_size; i++) {
        workgroup_data.items[i * workgroup_size + local_invocation_index] = precomp.items[i * workgroup_size + local_invocation_index];
    }
    if (precomp_size % workgroup_size != 0 && local_invocation_index < precomp_size % workgroup_size) {
        workgroup_data.items[precomp_size / workgroup_size * workgroup_size + local_invocation_index] = precomp.items[precomp_size / workgroup_size * workgroup_size + local_invocation_index];
    }
    workgroupBarrier();

    let index = (workgroup_id.y * num_workgroups.x + workgroup_id.x) * workgroup_size + local_invocation_index;

    let input_index = index >> 16;
    let seed_hi = index & 0xFFFF;

    if (input_index >= inputs.count) {
        return;
    }
    let input = inputs.items[input_index];

    let seed = u64_add(input.structure_seed, u64(0, seed_hi << 16));

    let start_xrsr = xrsr_setFeatureSeed(seed, (input.packed_chunk_x >> 16) << 4, (input.packed_chunk_z >> 16) << 4, 40019);

    var xrsr = u128(u64(0, 0), u64(0, 0));

__SKIP__

    for (var i = 0; i < 12; i++) {
        let long = xrsr_long(xrsr);
        if (long.hi < 0xe6666700) {
            return;
        }
        xrsr = xrsr_next(xrsr);
    }

    let result_index = atomicAdd(&results.count, 1);
    if (result_index >= arrayLength(&results.items)) {
        return;
    }
    results.items[result_index] = Result(seed, input.packed_chunk_x, input.packed_chunk_z);
}