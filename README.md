### Download
```bash
git clone https://github.com/Gaider10/12-eye
cd 12-eye
git submodule update --init
```

### Build a cuda executable
```bash
make main
```
`main[.exe]`

### Build for wasm (needs activated emsdk)
```bash
make wasm
```
Serve `web/src` through a web server

### Build a standalone .html file
```bash
make wasm-single
cd web
npm i
npm run build
```
`web/dist/index.html`