const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const HTMLInlineCSSWebpackPlugin = require('html-inline-css-webpack-plugin').default;
const HtmlInlineScriptPlugin = require('html-inline-script-webpack-plugin');

function rel(...paths) {
    return path.resolve(__dirname, ...paths);
}

const ifdefLoaderOptions = {
    'ifdef-triple-slash': false,
    'ifdef-fill-with-blanks': true,
    'ifdef-uncomment-prefix': '// #code ',
};

module.exports = {
    mode: 'production',
    entry: rel('src/main.js'),
    output: {
        filename: 'main.js',
        path: rel('dist'),
        asyncChunks: false,
    },
    resolve: {
        alias: {
            [rel('src/wasm/dist_worker.js')]: rel('dist/worker.js'),
        },
    },
    plugins: [
        new MiniCssExtractPlugin(),
        new HtmlWebpackPlugin({
            template: rel('src/index.html'),
        }),
        new HTMLInlineCSSWebpackPlugin(),
        new HtmlInlineScriptPlugin(),
    ],
    module: {
        rules: [
            {
                test: /\.css$/,
                use: [MiniCssExtractPlugin.loader, 'css-loader'],
            },
            {
                test: rel('dist/worker.js'),
                type: 'asset/inline',
            },
            {
                test: rel('src/webgpu/main.wgsl'),
                type: 'asset/source',
            },
            {
                test: /\.html$/,
                use: ['html-loader'],
            },
            {
                test: /\.m?js$|\.html$/,
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