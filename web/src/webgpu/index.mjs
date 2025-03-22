if (!navigator.gpu) throw new Error("WebGPU not supported");

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("Could not get GPU adapter");
console.log("GPU Adapter:", adapter);

const device = await adapter.requestDevice({ requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize }});
console.log("GPU Device:", device);

// #if false
const wgsl = await fetch(new URL("./main.wgsl", import.meta.url)).then((response) => response.text());
// #else
// #code import wgsl from './main.wgsl';
// #endif

// console.log("Wgsl:\n" + wgsl);

const precompBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "read-only-storage",
            },
        },
    ],
});

const dataBindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "read-only-storage",
            },
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "storage",
            },
        },
    ],
});

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
        precompBindGroupLayout,
        dataBindGroupLayout,
    ],
});

/**
 * @typedef {[number, number, number, number]} Xrsr
 */

/** @type {Xrsr[]} */
const SKIP_DATA = [
    [0x4956cf87, 0x9729b1f3, 0x235828b3, 0xbea1cb0e],
    [0xe6e4137b, 0x69438152, 0x37b8cc30, 0x30d9b3a6],
    [0x544d7f0d, 0x54d7aa11, 0x1df9826a, 0xef9ba2a9],
    [0x0f183868, 0x184dd269, 0x3f77b567, 0x6b77a2a2],
    [0xcd298822, 0x66cb946d, 0x8680083a, 0x8ee1a047],
    [0x782b3af2, 0x715a6f94, 0xbb0fca9a, 0x7396c0e3],
    [0x1d13dd35, 0xa57b30c7, 0x6709c14a, 0x9386458f],
    [0x6f353f18, 0x86075716, 0xbf1381b0, 0x77b6ca6d],
    [0x4773c0f0, 0x2eb84f43, 0x0e68ee6b, 0xe98f1e3a],
    [0xd3f5e13a, 0x6ec4a811, 0x81412058, 0x48a8feb4],
    [0xbb763692, 0x310cb9f1, 0xd1528435, 0x2acb074f],
    [0x0b269e02, 0x54445aa2, 0x93741e1e, 0x07cfecc8],
    [0xaf9dfdb0, 0xe9066c93, 0xf77a39ad, 0x175f9549],
    [0xe0d69c9b, 0x62c2b5e8, 0x54682e38, 0x4426c489],
    [0x0ed35ea9, 0x5e5ef7e0, 0xc90ce28c, 0x06eeb208],
    [0x522ae01b, 0xb1b3fcfb, 0x41018041, 0xb20ad128],
    [0xb6da0494, 0xfc7cf254, 0x0cc9cfd9, 0xf94466f6],
    [0xd4f693c4, 0xe6ee807d, 0x650b4e03, 0x7420dd36],
    [0xf7190436, 0x7ae8e208, 0x42496c56, 0xa4d5e848],
    [0x107b32c9, 0xd1ea97de, 0x513089ca, 0x0a1651d2],
    [0x283d5e3b, 0x5f5291a6, 0x1850a9a3, 0xfa7bfb68],
    [0xbb0cd1a4, 0xe075e04e, 0xf32c1b27, 0xef063377],
    [0x864c26ab, 0x4a94290c, 0x1f9ac0d7, 0x8ee41520],
    [0x3ab76ecc, 0x5e5ddf9a, 0xecd74a35, 0x59d29bb6],
    [0x4f4296e2, 0x80b8f5bd, 0x62ede900, 0x70d37d80],
    [0x5d7b1bf1, 0x14babf1d, 0xd103a272, 0xc325d770],
    [0x6f7362b3, 0xfe7068c5, 0x7be907e6, 0x1386ef19],
    [0x9a7949c0, 0xd22c0445, 0xc90a9cf1, 0x378500ec],
    [0x8cfa46df, 0x45f33f67, 0xeaa81ae4, 0xfe49609d],
    [0x7a964ff5, 0x0cbf3a8b, 0x835c4cdc, 0xc8df216f],
    [0xd68846d9, 0xb0c0a793, 0xaf5a1ec9, 0xadf96f98],
    [0xa04a5b72, 0x148a61fa, 0x7c709c9b, 0x814f2528],
    [0x5dd0442f, 0xa500b82a, 0xcb8e09f0, 0xe0165b6f],
    [0xa43a43fc, 0x4ab3bd9e, 0xba91e661, 0xa559a881],
    [0x69b47539, 0x10a26e20, 0x4bf3b1e7, 0x828686ae],
    [0x0d9126f2, 0x6617a247, 0xc3798857, 0x354a4970],
    [0xf71d7799, 0xc8cb1f7f, 0xaeff223b, 0x627399b8],
    [0x4650b5b4, 0xf5da697d, 0x9583aab8, 0xf5dc4894],
    [0x8e1f2c0a, 0xaed50dfa, 0x799464a4, 0x20b68ce2],
    [0x59224c5d, 0x00621d33, 0x57508398, 0x6ce16fa5],
    [0x187ad8cd, 0xdd0d3cab, 0x460fb080, 0xf5437e0d],
    [0xeca8340a, 0x0a7dff3c, 0x5179eb67, 0x5a678462],
    [0x8ae30744, 0x66c0ea49, 0xd37bf1da, 0x5bada798],
    [0x2a7c0f4d, 0x9649d079, 0xaf8b828d, 0xf49f3202],
    [0x0df2c4d7, 0xdc21a06b, 0xdfd5f696, 0x06a6f88e],
    [0x58eb5ff4, 0x5c369494, 0xf8755f8e, 0xa958de14],
    [0xd8980411, 0x67840ae2, 0x37852aca, 0xf41b985d],
    [0x32d607c3, 0x1002322c, 0x540cb5fc, 0x0889eabc],
    [0x396c6752, 0xd08621bb, 0x481d069a, 0x2df8ffea],
    [0xfbcf0701, 0xd677a3a5, 0x502fcc2e, 0x323e4d1e],
    [0x1fb583e8, 0x8fbc61e1, 0x4f1e28d6, 0x2a1b5a01],
    [0xf04b9382, 0x1eb45493, 0xfc2295fe, 0x79920c49],
    [0xe0b7274e, 0xd5706cc1, 0x77654919, 0xa6f2980b],
    [0x0da195ee, 0xbaae62ca, 0xf0d38bc5, 0x49242973],
    [0x62376218, 0x9a1086f1, 0x775c1cf5, 0x5bb41874],
    [0x6d5cecef, 0x5182ac80, 0xa00c0d56, 0xa1c27475],
    [0x31097121, 0xfb2ef045, 0xca3bf283, 0xfa8abd84],
    [0x3d689655, 0xac19179c, 0x9cb60c51, 0xca58f43d],
    [0xd7af7acc, 0xed43af5d, 0xcdea0d50, 0x4a258cc0],
    [0xb2d5f96c, 0x468128ef, 0x52f2c042, 0xcc1823d5],
    [0x86305762, 0xb25a4a53, 0x60d7a055, 0xedc3c15e],
    [0x27c38d24, 0x29041dcb, 0xd022f75f, 0x9b3edfeb],
    [0x2757e8c5, 0xc6443538, 0x8d70845a, 0xb4a161f0],
    [0x86988ee9, 0x66d82098, 0x5753c1e6, 0x1476f5e5],
    [0xa2248bf0, 0x19c7c3b6, 0xebcebd21, 0x32b37ba8],
    [0xf71a7a7e, 0x04585f12, 0x87f30521, 0xac20e084],
    [0xffc3ef3a, 0x4361d754, 0x2f57d6c2, 0x3a04800b],
    [0x5ce4f05c, 0x94781ed6, 0x03a2fde4, 0x9116ff87],
    [0x4c4116a0, 0x2dada86c, 0x95aa0187, 0xc8bf2012],
    [0x0a84b841, 0x6190775f, 0x72e1cf6e, 0x169d177d],
    [0xc8c92335, 0x5bfa7dda, 0xaef47810, 0x6dc7ddf8],
    [0x3b8390aa, 0xe0879f07, 0x430ef64e, 0xb2892a43],
    [0xadb5a286, 0x9f0c3ea0, 0x4121cfc4, 0x935baac5],
    [0x0895a881, 0xf664d22a, 0xf879a5ef, 0x0ea8518b],
    [0x596ac088, 0xbf4ed50a, 0xf2cc4f7a, 0x8ba56f98],
    [0x2a64947f, 0xb003a9c4, 0x6e0a0185, 0xeef21af0],
    [0x965137be, 0x63c0cd54, 0x7d5b0d1e, 0x9fb64177],
    [0x1b304e89, 0xfbeaf3c8, 0xe598e322, 0x0a06c702],
    [0xe68d314f, 0x37d7f24b, 0x663dd33f, 0x1a5ba9c2],
    [0x6d9802f0, 0x8d1ca37e, 0x94afd248, 0xbb270596],
    [0x22563dc3, 0x84e7a1db, 0x95decced, 0xfa53f67a],
    [0x7e38bc8c, 0xbb13ff96, 0x367f4bda, 0xf6458473],
    [0xfe3c94c6, 0x4f459c09, 0xfc16d685, 0x6603c03f],
    [0xd6560f22, 0xbba841c7, 0x7907b059, 0x08540463],
    [0xb259a49b, 0xaa8bd2b5, 0xbb585d38, 0xbf5e11dc],
    [0x79ed71c4, 0xf4077d95, 0x09a9e3b1, 0x8be8ba05],
    [0x4aadad43, 0x5a56ce90, 0xa817ede4, 0x6f1ee9bb],
    [0x2b1f613e, 0x85bd0a5b, 0x658ea667, 0x64bca8c9],
    [0x4a7bf3c1, 0x7e8ced80, 0xed5b676b, 0x9af56039],
    [0x7c4cbcc9, 0xa2ec7b2e, 0x5c8471b7, 0x605b5011],
    [0x4636cd36, 0xec3d9bb9, 0x2e5264ea, 0x368afd85],
    [0xceeeab85, 0x76982087, 0xe8642a2b, 0xe492beff],
    [0xe6fd6cbb, 0x4801e76a, 0xc2c37815, 0xb5395b14],
    [0x4584a4bb, 0x8d9d0257, 0xd2a5871e, 0xd9333173],
    [0x2fc7d7be, 0x4661b690, 0x76b89d2a, 0x4b407a80],
    [0x95b01afb, 0x570906b4, 0xddccb272, 0x0e50bff9],
    [0xfd811c76, 0xb5f8c537, 0x29d282cf, 0xa05ac645],
    [0x64633a67, 0x8ceaa8fd, 0x37961b7e, 0x47967888],
    [0x7e425e42, 0x034ce4ff, 0xfd6087b9, 0x826090d1],
    [0x79cc6824, 0x18200d96, 0x109b7000, 0xce777c9e],
    [0x0470b92c, 0xe7e04d38, 0xadf1dbf1, 0x7f997981],
    [0xfc05bba6, 0x03f57d6b, 0xc197f6ab, 0xfa5f51fa],
    [0xe4ce50c7, 0xc22968d5, 0x8e6ca307, 0x813e4a98],
    [0xfe501b1e, 0xe1c2bc69, 0xfd150cdd, 0xa39f5ef8],
    [0x2e04a84e, 0xb4c2fbe8, 0x422cb70e, 0x8cb69e62],
    [0x253a7269, 0x87c0cdc2, 0xa0855c93, 0x694ac8fd],
    [0x29655731, 0x066b779d, 0x08adaf63, 0xabaff55f],
    [0x0dfc9c0a, 0x97bd79af, 0x6c9d1814, 0x5fea359b],
    [0x217404a2, 0xf8a33459, 0x7ea1201b, 0x17e0a170],
    [0x0742b4e9, 0x00dd88bb, 0x784a4d19, 0x00122ea1],
    [0xdc62f651, 0xfc8078ce, 0x278d8a0d, 0xa81623c9],
    [0xcbbff9d3, 0x15cb73b6, 0xac00fd10, 0xb5a03428],
    [0x5c9ff721, 0x64ecd8db, 0x78741ef2, 0x308c6090],
    [0xf421b74f, 0xe8dd2d0b, 0x77a91684, 0xb3afddc8],
    [0xc74afc1a, 0x566f3dbb, 0xc345c028, 0xc4508328],
    [0x86423b7c, 0xf3490421, 0xad12a68a, 0x7b9d7f4d],
    [0xfec9c94b, 0x6841f59b, 0x34df2e55, 0x43f9aade],
    [0xfcb72a07, 0xaffec201, 0x0aadfa0b, 0xc532c3fa],
    [0x393340c1, 0x6925e83b, 0xe24ec479, 0xdd894f38],
    [0xf15032e1, 0x08ae87e6, 0x6837338c, 0x9815571c],
    [0x081e988a, 0x380e377d, 0x55545d0d, 0x2e34ccc9],
    [0x820c64b1, 0xfa489750, 0x99088738, 0xabb66954],
    [0x401cc040, 0x8e229c3a, 0x13aa97f6, 0xd1fc9778],
    [0x14c8ffd5, 0x84fc15f1, 0x9e804e16, 0xf33a77ab],
    [0xd96db47a, 0x5cbb3810, 0xaa07a202, 0x84774bf7],
    [0xe722c83b, 0x61ba8c6b, 0x517a3793, 0xabd2e2e4],
    [0x7154bca9, 0xf958d44c, 0x53ed7bf3, 0xcb76df44],
    [0x4a63bc6f, 0xba2b39a9, 0x44270dc4, 0x49558576],
];

/**
 * @param {Xrsr} a
 * @param {Xrsr} b
 * @returns Xrsr
 */
function XOR(a, b) {
    for (let i = 0; i < 4; i++) {
        a[i] ^= b[i];
    }
}

/**
 * @param {Xrsr} xrsr
 */
function skip(xrsr) {
    /** @type {Xrsr} */
    let out = [0, 0, 0, 0];
    for (let word = 0; word < 4; word++) {
        for (let bit = 0; bit < 32; bit++) {
            if (xrsr[word] >> bit & 1) {
                XOR(out, SKIP_DATA[word * 32 + bit]);
            }
        }
    }
    return out;
}

/**
 * @param {Uint32Array} table
 * @param {number} first
 * @param {number} count
 */
function precompute_bits(table, word, first, count) {
    /** @type {Xrsr} */
    let xrsr = [0, 0, 0, 0];

    for (let bits = 0; bits < 2**count; bits++) {
        xrsr[word] = bits << first;
        table.set(skip(xrsr), bits * 4);
    }
}

/**
 * @typedef {[word: number, bit: number, size: number][]} PrecompRanges
 * @typedef {[size: number, count: number][]} SimplePrecompRanges
 */

/**
 * @param {SimplePrecompRanges} counts
 * @returns {PrecompRanges}
 */
export function makeRanges(counts) {
    /** @type {[word: number, bit: number, size: number][]} */
    const ranges = [];

    let word = 0;
    let bit = 0;
    for (const [size, count] of counts) {
        for (let i = 0; i < count; i++) {
            if (bit + size > 32) {
                word += 1;
                bit = 0;
                if (word >= 4) {
                    throw new Error("Ranges don't fit");
                }
            }
            ranges.push([word, bit, size]);
            bit += size;
        }
    }

    return ranges;
}

const workgroupSize = 256;

export const MAX_INPUTS = 2**16;
export const MAX_OUTPUTS = 1024;
const PRECOMP_ITEM_SIZE = 16;
const INPUT_ITEM_SIZE = 16;
const OUTPUT_ITEM_SIZE = 16;
const INPUT_BUFFER_SIZE = 4 + INPUT_ITEM_SIZE * MAX_INPUTS;
const OUTPUT_BUFFER_SIZE = 4 + OUTPUT_ITEM_SIZE * MAX_OUTPUTS;

/**
 * @param {number} totalInvocations
 * @returns {[workgroupCountX: number, workgroupCountY: number, workgroupCountZ: number]}
 */
function calcWorkgroupCounts(totalInvocations) {
    const maxCount = device.limits.maxComputeWorkgroupsPerDimension;
    const workgroupCountX = Math.min(Math.ceil(totalInvocations / (workgroupSize)), maxCount);
    const workgroupCountY = Math.min(Math.ceil(totalInvocations / (workgroupSize * workgroupCountX)), maxCount);
    const workgroupCountZ = Math.min(Math.ceil(totalInvocations / (workgroupSize * workgroupCountX * workgroupCountY)), maxCount);
    if (workgroupCountZ !== 1) throw new Error("workgroupCountZ must be 1");
    return [workgroupCountX, workgroupCountY, workgroupCountZ]
}

export class DataBuffers {
    /** @type {Uint32Array} */
    inputHostBuffer
    /** @type {Uint32Array} */
    outputHostBuffer
    /** @type {GPUBuffer} */
    inputBuffer
    /** @type {GPUBuffer} */
    outputBuffer
    /** @type {GPUBuffer} */
    outputStagingBuffer
    /** @type {GPUBindGroup} */
    dataBindGroup

    constructor() {
        const inputHostBuffer = new Uint32Array(INPUT_BUFFER_SIZE / 4);
        const outputHostBuffer = new Uint32Array(OUTPUT_BUFFER_SIZE / 4);

        const inputBuffer = device.createBuffer({
            size: INPUT_BUFFER_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const outputBuffer = device.createBuffer({
            size: OUTPUT_BUFFER_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const outputStagingBuffer = device.createBuffer({
            size: OUTPUT_BUFFER_SIZE,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const dataBindGroup = device.createBindGroup({
            layout: dataBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: inputBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: outputBuffer,
                    },
                },
            ],
        });

        this.inputHostBuffer = inputHostBuffer;
        this.outputHostBuffer = outputHostBuffer;
        this.inputBuffer = inputBuffer;
        this.outputBuffer = outputBuffer;
        this.outputStagingBuffer = outputStagingBuffer;
        this.dataBindGroup = dataBindGroup;
    }
}

export class Kernel {
    /** @type {GPUBindGroup} */
    precompBindGroup
    /** @type {Promise<GPUComputePipeline>} */
    computePipelinePromise

    /**
     * @param {PrecompRanges} precompRanges
     */
    constructor(precompRanges = makeRanges([[5, 20]])) {
        let precompSize = 0;
        let precompMask = [0, 0, 0, 0];

        for (const [word, first, count] of precompRanges) {
            if (first + count > 32) {
                throw new Error("Invalid precomp range");
            }

            const bit = first % 32;
            const mask = (2**count - 1) << bit;
            if ((precompMask[word] & mask) != 0) {
                throw new Error("Overlapping precomp ranges");
            }
            precompMask[word] |= mask;

            precompSize += 2**count;
        }
        if (precompSize === 0) precompSize = 1;

        const precompBufferSize = PRECOMP_ITEM_SIZE * precompSize;
        const precompHostBuffer = new Uint32Array(precompBufferSize / 4);
        const precompBuffer = device.createBuffer({
            size: precompBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const precompBindGroup = device.createBindGroup({
            layout: precompBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: precompBuffer,
                    },
                },
            ],
        });

        const WORD_PATHS = [
            ".lo.lo",
            ".lo.hi",
            ".hi.lo",
            ".hi.hi",
        ];

        const precompWgsl = [];
        let offset = 0;
        for (const [word, first, count] of precompRanges) {
            precompute_bits(precompHostBuffer.subarray(4 * offset), word, first, count);
            precompWgsl.push(`xrsr = u128_xor(xrsr, workgroup_data.items[${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})]);`);
            // precompWgsl.push(`xrsr = u128_xor(xrsr, workgroup_data.items[${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})].xrsr);`);
            // precompWgsl.push(`xrsr = u128_xor(xrsr, precomp.items[${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})]);`);
            // precompWgsl.push(`xrsr = u128_xor(xrsr, get_precomp(${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})));`);
            // precompWgsl.push(`{ let index = ${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1}); xrsr = u128_xor(xrsr, u128(u64(workgroup_data.items[index * 4 + 0], workgroup_data.items[index * 4 + 1]), u64(workgroup_data.items[index * 4 + 2], workgroup_data.items[index * 4 + 3]))); }`);
            // precompWgsl.push(`{ let index = ${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1}); xrsr = u128_xor(xrsr, u128(u64(workgroup_data.items[index * 4 + 3], workgroup_data.items[index * 4 + 2]), u64(workgroup_data.items[index * 4 + 1], workgroup_data.items[index * 4 + 0]))); }`);
            // precompWgsl.push(`{ let index = ${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1}); xrsr.lo.lo ^= workgroup_data.items[index * 4 + 0]; xrsr.lo.hi ^= workgroup_data.items[index * 4 + 1]; xrsr.hi.lo ^= workgroup_data.items[index * 4 + 2]; xrsr.hi.hi ^= workgroup_data.items[index * 4 + 3]; }`);
            // precompWgsl.push(`{ let index = (${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})) * 5; xrsr.lo.lo ^= workgroup_data.items[index + 0]; xrsr.lo.hi ^= workgroup_data.items[index + 1]; xrsr.hi.lo ^= workgroup_data.items[index + 2]; xrsr.hi.hi ^= workgroup_data.items[index + 3]; }`);
            // precompWgsl.push(`{ let index = (${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})) * 3; xrsr.lo.lo ^= workgroup_data.items[index + 0]; xrsr.lo.hi ^= workgroup_data.items[index + 32]; xrsr.hi.lo ^= workgroup_data.items[index + 64]; xrsr.hi.hi ^= workgroup_data.items[index + 96]; }`);
            // precompWgsl.push(`{ let index = (${offset} + ((start_xrsr${WORD_PATHS[word]} >> ${first}) & ${2**count - 1})); xrsr.lo.lo ^= workgroup_data.items[index].xrsr.lo.hi; xrsr.lo.hi ^= workgroup_data.items[index].xrsr.lo.lo; xrsr.hi.lo ^= workgroup_data.items[index].xrsr.hi.lo; xrsr.hi.hi ^= workgroup_data.items[index].xrsr.hi.hi; }`);

            offset += 2**count;
        }

        device.queue.writeBuffer(precompBuffer, 0, precompHostBuffer);

        const skipWgsl = [];
        for (let word = 0; word < 4; word++) {
            for (let bit = 0; bit < 32; bit++) {
                if (!(precompMask[word] >> bit & 1)) {
                    skipWgsl.push(`if (bool((start_xrsr${WORD_PATHS[word]} >> ${bit}) & 1)) { xrsr = u128_xor(xrsr, u128(u64(${SKIP_DATA[word * 32 + bit][0]}, ${SKIP_DATA[word * 32 + bit][1]}), u64(${SKIP_DATA[word * 32 + bit][2]}, ${SKIP_DATA[word * 32 + bit][3]}))); }`);
                }
            }
        }

        const mergedWgsl = [];
        for (let i = 0; i < precompWgsl.length; i++) {
            mergedWgsl.push(precompWgsl[i]);

            const start = Math.floor(skipWgsl.length * i / precompWgsl.length);
            const end = Math.floor(skipWgsl.length * (i + 1) / precompWgsl.length);
            mergedWgsl.push(...skipWgsl.slice(start, end));
        }
        if (precompWgsl.length === 0) {
            mergedWgsl.push(...skipWgsl);
        }
        // for (let i = precompWgsl.length; i >= 2; i--) {
        //     const j = Math.floor(Math.random() * i);
        //     [mergedWgsl[i], mergedWgsl[j]] = [mergedWgsl[j], mergedWgsl[i]];
        // }
        console.log(skipWgsl, precompWgsl);

        // mergedWgsl.length = 0;
        // mergedWgsl.push(...precompWgsl, ...skipWgsl);

        const newWgsl = wgsl
            .replace("__SKIP__", mergedWgsl.map(s => `    ${s}`).join("\n"))
            .replace("__PRECOMP_SIZE__", String(precompSize))
            .replace("__WORKGROUP_SIZE__", String(workgroupSize));
        console.log(newWgsl);

        const shaderModule = device.createShaderModule({
            code: newWgsl,
        });

        const computePipelinePromise = device.createComputePipelineAsync({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });

        this.precompBindGroup = precompBindGroup;
        this.computePipelinePromise = computePipelinePromise;
    }

    /**
     * @param {DataBuffers} dataBuffers
     */
    async run(dataBuffers) {
        const computePipeline = await this.computePipelinePromise;

        const inputsCount = dataBuffers.inputHostBuffer[0];
        if (inputsCount > MAX_INPUTS) throw new Error(`inputsCount > MAX_INPUTS: ${inputsCount} > ${MAX_INPUTS}`);

        const totalInvocations = inputsCount * 2**16;
        if (totalInvocations > 2**32) throw new Error(`totalInvocationns > 2**32: ${totalInvocations}`);

        const workgroupCounts = calcWorkgroupCounts(totalInvocations)
        // console.log(`workgroupCounts = ${workgroupCounts.join(" ")}`);

        device.queue.writeBuffer(dataBuffers.inputBuffer, 0, dataBuffers.inputHostBuffer);
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.clearBuffer(dataBuffers.outputBuffer, 0, 4);
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, this.precompBindGroup);
        passEncoder.setBindGroup(1, dataBuffers.dataBindGroup);
        passEncoder.dispatchWorkgroups(...workgroupCounts);
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(dataBuffers.outputBuffer, 0, dataBuffers.outputStagingBuffer, 0, OUTPUT_BUFFER_SIZE);
        device.queue.submit([commandEncoder.finish()]);

        await dataBuffers.outputStagingBuffer.mapAsync(GPUMapMode.READ, 0, OUTPUT_BUFFER_SIZE);

        dataBuffers.outputHostBuffer.set(new Uint32Array(dataBuffers.outputStagingBuffer.getMappedRange(0, OUTPUT_BUFFER_SIZE)));

        dataBuffers.outputStagingBuffer.unmap();
    }
}

export {};