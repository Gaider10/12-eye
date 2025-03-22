// #if false
import CreateModule from './lib.mjs';
// #else
// #code import CreateModule from './lib_single.mjs';
// #endif

/**
 * @type {EmscriptenModule & {
 *   _generate_layouts: (structure_seed_start: bigint, structure_seed_end: bigint, out: number, out_len: number) => number,
 *   _test_world_seed: (world_seed: bigint, start_chunk_x: number, start_chunk_z: number) => number,
 * }}
 */
const Module = await CreateModule();

/**
 * @param {bigint} structure_seed_start
 * @param {bigint} structure_seed_end
 * @returns {Uint32Array}
 */
function generate_layouts(structure_seed_start, structure_seed_end) {
    const out_len = Number(structure_seed_end - structure_seed_start) + 64;
    const out = Module._malloc(16 * out_len);
    const count = Module._generate_layouts(structure_seed_start, structure_seed_end, out, out_len);
    const res = Module.HEAPU32.slice(out / 4, out / 4 + count * 4);
    Module._free(out);
    return res;
}

/**
 * @param {bigint} world_seed
 * @param {number} start_chunk_x
 * @param {number} start_chunk_z
 * @returns {boolean}
 */
function test_world_seed(world_seed, start_chunk_x, start_chunk_z) {
    return Module._test_world_seed(world_seed, start_chunk_x, start_chunk_z) !== 0;
}

function log(data) {
    self.postMessage({ call: 'log', data });
}

function bench_layout() {
    log(`Hello!`);
    const seeds_per_run = 2**16;

    let start = performance.now();

    for (let i = 0;; i++) {
        const out = generate_layouts(BigInt(i * seeds_per_run), BigInt((i + 1) * seeds_per_run));
        // log(`${count} / ${out_len}`);

        const print_interval = 1;
        const new_i = i + 1;
        if (new_i % print_interval === 0) {
            const end = performance.now();
            const delta = (end - start) * 1E-3;
            const per_sec = print_interval * seeds_per_run / delta;
            // console.log(`${new_i} ${delta.toFixed(3)} s ${per_sec.toFixed(3)} sps`);
            log(`${new_i} ${delta.toFixed(3)} s ${per_sec.toFixed(3)} sps`);
            start = end;
        }
    }
}

function bench_biomes() {
    let start = performance.now();

    for (let i = 0;; i++) {
        test_world_seed(BigInt(i), -30, -164);

        const print_interval = 50;
        const new_i = i + 1;
        if (new_i % print_interval === 0) {
            const end = performance.now();
            const delta = (end - start) * 1E-3;
            const per_sec = print_interval / delta;
            log(`${new_i} ${delta.toFixed(3)} s ${per_sec.toFixed(3)} sps`);
            start = end;
        }
    }
}

// console.log(Module._test_world_seed(123n, -23, 45));
// console.log(Module._test_world_seed(123n, 124, 49));
// console.log(Module._test_world_seed(123n, -116, 102));
// console.log(Module._test_world_seed(123n, -30, -164));

const CALLS = {
    'bench_layout': (data) => {
        bench_layout();
    },
    'bench_biomes': (data) => {
        bench_biomes();
    },
    /**
     * @param {{ id: any, data: { structure_seed_start: bigint, structure_seed_end: bigint }}} param0
     */
    'generate_layouts': ({ id, data: { structure_seed_start, structure_seed_end }}) => {
        const out = generate_layouts(structure_seed_start, structure_seed_end);
        self.postMessage({ id, data: out }, [out.buffer]);
    },
    /**
     * @param {{ id: any, data: { world_seed: bigint, start_chunk_x: number, start_chunk_z: number }}} param0
     */
    'test_world_seed': ({ id, data: { world_seed, start_chunk_x, start_chunk_z }}) => {
        self.postMessage({ id, data: test_world_seed(world_seed, start_chunk_x, start_chunk_z) });
    },
};

self.onmessage = (event) => {
    // console.log('Worker received:', event);

    CALLS[event.data?.call]?.(event.data);
};

self.postMessage(null);