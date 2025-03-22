// #if true
// #code import './style.css';
// #endif

import { DataBuffers, Kernel, makeRanges } from './webgpu/index.mjs';
import { benchBiomes, benchLayout, WorkerPool } from './wasm/index.mjs'

/**
 * @param {string} text
 * @param {string} [style]
 */
function log(text, style) {
    const div = document.createElement('div');
    div.innerText = text;
    if (style !== undefined) div.style = style;
    document.getElementById('div-log')?.appendChild(div);
}

/**
 * @overload
 * @param {HTMLInputElement} element
 * @param {number} min
 * @param {number} max
 * @param {number} def
 * @returns {{
 *     element: HTMLInputElement,
 *     value: number,
 * }}
 * @overload
 * @param {HTMLInputElement} element
 * @param {number} min
 * @param {number} max
 * @param {number | null} def
 * @param {true} nullable
 * @returns {{
 *     element: HTMLInputElement,
 *     value: number | null,
 * }}
 * /**
 * @param {HTMLInputElement} element
 * @param {number} min
 * @param {number} max
 * @param {number | null} def
 * @param {boolean} [nullable]
 */
function numberInput(element, min, max, def, nullable) {
    let val;
    const obj = {
        element,
        set value(value) {
            val = typeof value !== "number" || Number.isNaN(value) ? (nullable ? null : def) : Math.min(Math.max(Math.round(Number(value)), min), max);
            this.element.value = val === null ? '' : String(val);
        },
        get value() {
            return val;
        },
    };
    element.addEventListener('change', () => {
        obj.value = element.value === '' ? null : Number(element.value);
    });
    obj.value = def;
    return obj;
}

const inputStartSeed = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-start-seed')), 0, 2**48 - 1, null, true);
const inputBitsPerIter = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-bits-per-iter')), 1, 16, 16);
const inputWorkerCount = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-workers')), 1, 256, navigator.hardwareConcurrency);
const inputRangesSize = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-ranges-size')), 1, 10, 5);
const inputRangesCount = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-ranges-count')), 0, 50, 20);
const inputPrintInterval = numberInput(/** @type {HTMLInputElement} */ (document.getElementById('input-print-interval')), 0, 2**16, 1);

const INPUT_VALUES_KEY = 'inputValues';

function loadInputValues() {
    const storedInputValues = JSON.parse(localStorage.getItem(INPUT_VALUES_KEY) ?? '{}');

    if (storedInputValues?.startSeed !== undefined) inputStartSeed.value = storedInputValues?.startSeed;
    if (storedInputValues?.bitsPerIter !== undefined) inputBitsPerIter.value = storedInputValues?.bitsPerIter;
    if (storedInputValues?.workerCount !== undefined) inputWorkerCount.value = storedInputValues?.workerCount;
    if (storedInputValues?.rangesSize !== undefined) inputRangesSize.value = storedInputValues?.rangesSize;
    if (storedInputValues?.rangesCount !== undefined) inputRangesCount.value = storedInputValues?.rangesCount;
    if (storedInputValues?.printInterval !== undefined) inputPrintInterval.value = storedInputValues?.printInterval;
}

function saveInputValues() {
    localStorage.setItem(INPUT_VALUES_KEY, JSON.stringify({
        startSeed: inputStartSeed.value,
        bitsPerIter: inputBitsPerIter.value,
        workerCount: inputWorkerCount.value,
        rangesSize: inputRangesSize.value,
        rangesCount: inputRangesCount.value,
        printInterval: inputPrintInterval.value,
    }));
}

loadInputValues();

const divStatus = /** @type {HTMLDivElement} */ (document.getElementById('div-status'));

let stopGpuFlag = false;

async function runGPU() {
    stopGpuFlag = false;

    const bitsPerIter = inputBitsPerIter.value;
    const unmaskedStartStructureSeed = inputStartSeed.value !== null ? inputStartSeed.value : Math.floor(Math.random() * 2**48);
    const startStructureSeed = Math.floor(unmaskedStartStructureSeed / 2**bitsPerIter) * 2**bitsPerIter;

    inputStartSeed.value = startStructureSeed;
    saveInputValues();

    log(`Starting from ${startStructureSeed}`);
    // log(`Using ranges ${JSON.stringify(ranges)}`);

    const workerPool = new WorkerPool();
    const dataBufferSets = [0, 0].map(() => new DataBuffers());
    const kernel = new Kernel(makeRanges([[inputRangesSize.value, inputRangesCount.value]]));

    let prevIterationDonePromise = Promise.resolve();
    let itersSinceLastPrint = 0;
    let timeGpu = 0;
    let start = performance.now();

    for (let iter = 0;; iter++) {
        const iterStartStructureSeed = (startStructureSeed + iter * 2**bitsPerIter) % 2**48;
        const dataBuffers = dataBufferSets[iter % dataBufferSets.length];
        if (stopGpuFlag) break;

        await workerPool.generateLayouts(inputWorkerCount.value, BigInt(iterStartStructureSeed), BigInt(2**bitsPerIter), dataBuffers.inputHostBuffer);

        const iterationDonePromise = kernel.run(dataBuffers).then(() => {
            const gpuEnd = performance.now();
            timeGpu += gpuEnd - gpuStart;

            const resultCount = dataBuffers.outputHostBuffer[0];
            for (let i = 0; i < resultCount; i++) {
                const seed = BigInt(dataBuffers.outputHostBuffer[1 + i * 4 + 0]) + (BigInt(dataBuffers.outputHostBuffer[1 + i * 4 + 1] | 0) << 32n);
                const startChunkX = dataBuffers.outputHostBuffer[1 + i * 4 + 2] << 16 >> 16;
                const portalChunkX = dataBuffers.outputHostBuffer[1 + i * 4 + 2] >> 16;
                const startChunkZ = dataBuffers.outputHostBuffer[1 + i * 4 + 3] << 16 >> 16;
                const portalChunkZ = dataBuffers.outputHostBuffer[1 + i * 4 + 3] >> 16;
                const I = i;
                workerPool.testWorldSeed(seed, startChunkX, startChunkZ).then((valid) => {
                    // log(`Seed: ${seed} Start: ${startChunkX} ${startChunkZ} Pos: ${portalChunkX * 16} ~ ${portalChunkZ * 16} Valid: ${valid ? 'YES' : 'no'}`);
                    if (valid) {
                        log(`Seed: ${seed} Start: ${startChunkX} ${startChunkZ} Pos: ${portalChunkX * 16} ~ ${portalChunkZ * 16}`, 'font-weight: bold;');
                        // log(`iterStartStructureSeed = ${iterStartStructureSeed} | resultCount = ${resultCount} | i = ${I}`);
                    }
                });
            }

            itersSinceLastPrint += 1;
            if (inputPrintInterval.value !== 0 && itersSinceLastPrint >= inputPrintInterval.value) {
                const end = performance.now();
                const elapsedMs = end - start;
                const elapsedS = elapsedMs * 1E-3;
                const seedsPerRun = 2**(bitsPerIter + 16);
                const sps = seedsPerRun * itersSinceLastPrint / elapsedS;
                // console.log(`Took ${elapsedMs.toFixed(3)} ms (${(timeGpu / elapsedMs * 100).toFixed(3)} % gpu), ${(sps * 1E-9).toFixed(3)} Gsps, at ${structureSeedHi + 1} / ${2**32}, ETA: ${((2**32 - (structureSeedHi + 1)) * 2**32 / sps / 3600).toFixed(1)} h`);
                // log(`Took ${elapsedMs.toFixed(3)} ms (${(timeGpu / elapsedMs * 100).toFixed(3)} % gpu), ${(sps * 1E-9).toFixed(3)} Gsps, at ${(iterStartStructureSeed + 2**bitsPerIter) % 2**48}`);
                const nextIterStartStructureSeed = (iterStartStructureSeed + 2**bitsPerIter) % 2**48;
                divStatus.innerText =
                    `Took ${elapsedMs.toFixed(3)} ms\n` +
                    `Speed: ${(sps * 1E-9).toFixed(3)} Gsps\n` +
                    `GPU utilization: ${(timeGpu / elapsedMs * 100).toFixed(3)} %\n` +
                    `Structure seed: ${nextIterStartStructureSeed}`;
                start = end;
                timeGpu = 0;
                itersSinceLastPrint = 0;
                inputStartSeed.value = nextIterStartStructureSeed;
                saveInputValues();
            }
        });

        await prevIterationDonePromise;
        const gpuStart = performance.now();

        prevIterationDonePromise = iterationDonePromise;
    }

    await prevIterationDonePromise;
}

const benchLayoutButton = /** @type {HTMLButtonElement} */ (document.getElementById("bench-layout-button"));
benchLayoutButton.addEventListener("click", () => {
    benchLayoutButton.disabled = true;
    benchLayout();
});

const benchBiomesButton = /** @type {HTMLButtonElement} */ (document.getElementById("bench-biomes-button"));
benchBiomesButton.addEventListener("click", () => {
    benchBiomesButton.disabled = true;
    benchBiomes();
});

const runGPUButton = /** @type {HTMLButtonElement} */ (document.getElementById("run-gpu-button"));
runGPUButton.addEventListener("click", () => {
    runGPUButton.disabled = true;
    stopGPUButton.disabled = false;
    inputStartSeed.element.disabled = true;
    inputBitsPerIter.element.disabled = true;
    inputRangesSize.element.disabled = true;
    inputRangesCount.element.disabled = true;
    runGPU().catch(e => log(`Error: ${e}`, 'color: red;')).finally(() => {
        log(`Stopped at ${inputStartSeed.value}`);
        runGPUButton.disabled = false
        stopGPUButton.disabled = true;
        inputStartSeed.element.disabled = false;
        inputBitsPerIter.element.disabled = false;
        inputRangesSize.element.disabled = false;
        inputRangesCount.element.disabled = false;
    });
});

const stopGPUButton = /** @type {HTMLButtonElement} */ (document.getElementById("stop-gpu-button"));
stopGPUButton.addEventListener("click", () => {
    stopGpuFlag = true;
});