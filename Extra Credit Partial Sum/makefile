CUDA_SRC 	= Partial_Sum.cu


OBJ_DIR		= obj
CPP_OBJ		= $(patsubst %.cpp,$(OBJ_DIR)/%.cpp.o,$(notdir $(CPP_SRC))) 
CUDA_OBJ	= $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CUDA_SRC)))


OBJ 		= $(CPP_OBJ) $(CUDA_OBJ) 

NVCC 		= /cm/shared/apps/cuda90/toolkit/9.0.176/bin/nvcc
#CC 		= /usr/local/gcc/5.4.0/bin/g++

PROFILER 	= /cm/shared/apps/cuda90/toolkit/9.0.176/bin/nvprof

INCLUDE 	= -I/usr/local/cuda/incldude
NVCC_FLAGS 	= --ptxas-options="-v" -O3 
#CFLAGS		= -03
LD_FLAGS 	= -L/usr/local/cuda/lib64 -lcudart -lm
EXE 		=  partial_sum


default: test makedirectories $(EXE)

test: makedirectories
	$(info $$CPP_OBJ is ${CPP_OBJ})
	$(info $$CUDA_OBJ is ${CUDA_OBJ})
	$(info $$OBJ is ${OBJ})

makescript: test ${EXE} 
	@echo ${PROFILER} ${CURDIR}/${EXE} > ${EXE}.sh	
		
tardis: makescript
	qsub -q tardis ${EXE}.sh | grep -oE "[0-9]+" > .tardis_qid
	
link: 
	rm -rf tardis.out
	rm -rf tardis.error
	ln -s ./${EXE}.sh.o$(shell cat .tardis_qid) tardis.out
	ln -s ./${EXE}.sh.e$(shell cat .tardis_qid) tardis.error

$(EXE): $(OBJ)
	$(NVCC) $(LD_FLAGS) $(OBJ) -o $@

$(OBJ_DIR)/%.cu.o: $(CUDA_SRC)
	$(NVCC) $(DEBUG) $(MACRO) $(INCLUDE) $(NVCC_FLAGS) -c $< -o $@

makedirectories: 
	$(VERBOSE)mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR)
	rm  $(EXE)
	rm  $(EXE).sh.*

