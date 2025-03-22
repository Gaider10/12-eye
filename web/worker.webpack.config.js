const path = require('path');

function rel(...paths) {
    return path.resolve(__dirname, ...paths);
}

const ifdefLoaderOptions = {
    'ifdef-triple-slash': false,
    'ifdef-uncomment-prefix': '// #code ',
};

module.exports = {
    mode: 'production',
    target: 'webworker',
    entry: rel('src/wasm/worker.mjs'),
    output: {
        filename: 'worker.js',
        path: rel('dist'),
        asyncChunks: false,
    },
    resolve: {
        alias: {
            module$: false,
        },
    },
    module: {
        rules: [
            {
                test: /\.m?js$/,
                include: rel('src/'),
                use: [{ loader: 'ifdef-loader', options: ifdefLoaderOptions }],
            },
        ],
        parser: {
            javascript: {
                importMeta: false,
            },
        },
    },
};