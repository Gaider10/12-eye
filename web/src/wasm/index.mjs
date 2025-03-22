// #if true
// #code import workerDataURI from './dist_worker.js';
// #endif

/**
 * @param {Worker} worker
 * @param {string} [call]
 * @param {any} [data]
 * @param {Transferable[]} [transfer]
 */
async function workerCall(worker, call, data, transfer) {
    if (worker.onmessage ?? worker.onmessageerror ?? worker.onerror ?? null !== null) {
        throw new Error('workerCall reentered');
    }

    const id = Math.random();

    return new Promise((resolve, reject) => {
        worker.onmessage = (event) => {
            if (call !== undefined && event.data?.id !== id) return;
            worker.onmessage = null;
            worker.onmessageerror = null;
            worker.onerror = null;
            resolve(event.data?.data);
        };
        worker.onmessageerror = (event) => {
            worker.onmessage = null;
            worker.onmessageerror = null;
            worker.onerror = null;
            reject(event);
        };
        worker.onerror = (event) => {
            worker.onmessage = null;
            worker.onmessageerror = null;
            worker.onerror = null;
            reject(event);
        };

        if (call !== undefined) {
            // console.log('Sending to worker:', { call, id, data });
            worker.postMessage({ call, id, data }, { transfer });
        }
    });
}

/**
 * @returns {Promise<Worker>}
 */
async function createWorker() {
    // #if false
    const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    // #else
    // #code const worker = new Worker(workerDataURI, { type: 'module' });
    // #endif
    worker.addEventListener('message', (event) => {
        // console.log('Received from worker:', event);
        if (event.data?.call === 'log') {
            const div = document.createElement('div');
            div.innerText = event.data.data;
            document.getElementById('div-log')?.appendChild(div);
        }
    });
    await workerCall(worker);
    return worker;
}

export function benchLayout() {
    createWorker().then(worker => {
        workerCall(worker, 'bench_layout')
            .finally(() => worker.terminate());
    });
}

export function benchBiomes() {
    createWorker().then(worker => {
        workerCall(worker, 'bench_biomes')
            .finally(() => worker.terminate());
    });
}

export class WorkerPool {
    /** @type {Promise<Worker>[]} */
    layoutWorkers;
    /** @type {Promise<Worker>} */
    biomeWorkerPromise;

    constructor() {
        this.layoutWorkers = [];
        this.biomeWorkerPromise = createWorker();
    }

    /**
     * @param {number} workerCount
     * @param {bigint} structureSeedStart
     * @param {bigint} structureSeedCount
     * @param {Uint32Array} out
     */
    async generateLayouts(workerCount, structureSeedStart, structureSeedCount, out) {
        if (this.layoutWorkers.length > workerCount) {
            this.layoutWorkers.splice(workerCount).forEach((workerPromise, i) => {
                // console.log(`Terminating layout worker ${i + this.layoutWorkers.length}`);
                workerPromise.then((worker) => worker.terminate());
            });
        }
        for (let i = this.layoutWorkers.length; i < workerCount; i++) {
            // console.log(`Creating layout worker ${i}`);
            this.layoutWorkers.push(createWorker());
        }

        const worker_outs = await Promise.all(this.layoutWorkers.map(async (workerPromise, i) => {
            const workerStructureSeedStart = structureSeedStart + structureSeedCount * BigInt(i) / BigInt(workerCount);
            const workerStructureSeedEnd = structureSeedStart + structureSeedCount * BigInt(i + 1) / BigInt(workerCount);

            const worker = await workerPromise;
            /** @type {Uint32Array} */
            const out = await workerCall(worker, 'generate_layouts', { structure_seed_start: workerStructureSeedStart, structure_seed_end: workerStructureSeedEnd });
            return out;
        }));

        const out_cap = Math.floor((out.length - 1) / 4);
        let out_len = 0;
        for (const worker_out of worker_outs) {
            const worker_out_len = Math.floor(worker_out.length / 4);
            const remaining_out_len = out_cap - out_len;
            if (worker_out_len > remaining_out_len) throw new Error(`Layout buffer too small`);
            const copied_len = Math.min(worker_out_len, remaining_out_len)
            out.set(worker_out.subarray(0, copied_len * 4), 1 + out_len * 4);
            out_len += copied_len;
        }
        out[0] = out_len;
    }

    /**
     * @param {bigint} worldSeed
     * @param {number} startChunkX
     * @param {number} startChunkZ
     * @returns {Promise<boolean>}
     */
    async testWorldSeed(worldSeed, startChunkX, startChunkZ) {
        while (true) {
            const workerPromise = this.biomeWorkerPromise;
            const worker = await workerPromise;
            if (this.biomeWorkerPromise !== workerPromise) continue;

            const callPromise = workerCall(worker, 'test_world_seed', { world_seed: worldSeed, start_chunk_x: startChunkX, start_chunk_z: startChunkZ });
            this.biomeWorkerPromise = callPromise.then(() => worker, () => worker);
            return await callPromise;
        }
    }
}