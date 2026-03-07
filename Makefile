CUBIOMES_SRC = $(addprefix cubiomes/,biomenoise.c biomes.c finders.c generator.c layers.c noise.c)
STRONGHOLD_GENERATOR_SRC = $(addprefix src/stronghold_generator/,BoundingBox.cpp Piece.cpp PiecePlaceCount.cpp PieceWeight.cpp StrongholdGenerator.cpp XrsrRandom.cpp)
LIB_SRC = src/lib.cpp src/lib.c $(STRONGHOLD_GENERATOR_SRC) $(CUBIOMES_SRC)

override NVCC_FLAGS += -std=c++20 -O3 --expt-relaxed-constexpr -arch=native --generate-line-info

ifeq ($(OS),Windows_NT)
	override NVCC_FLAGS += -Xcompiler /Zc:preprocessor -Xcompiler /GL
	EXE = .exe
endif

main: FORCE
	nvcc src/12eye.cu $(LIB_SRC) -o main$(EXE) $(NVCC_FLAGS)

WASM_EXPORTED = -sEXPORTED_FUNCTIONS=_malloc,_free,_generate_layouts,_test_world_seed

wasm: FORCE
	emcc $(LIB_SRC) -o web/src/wasm/lib.mjs -O3 $(WASM_EXPORTED)

wasm-single: FORCE
	emcc $(LIB_SRC) -o web/src/wasm/lib_single.mjs -O3 $(WASM_EXPORTED) -sSINGLE_FILE

FORCE: