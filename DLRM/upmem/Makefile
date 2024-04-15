# Setting defaults

PROJECT             = emb_host
PROJECT_LIB         = emblib.so
EXE_DPU             ?= emb_dpu_lookup
BUILD_DIR           ?= build
NR_DPUS             ?= 1024
NR_TASKLETS         ?= 14
COUNTER_CONFIG      ?= "COUNT_CYCLES"
SHOW_DPU_LOGS       ?= 1
MAX_ENC_BUFFER_MB   ?= 1
NR_TABLES           ?= 8
MAX_NR_BATCHES      ?= 32
RT_CONFIG           ?= "ALL"

# TEST with c_test.py
DPU_TEST            ?= 0

ifeq ($(DPU_TEST),1)
  NR_COLS             ?= 6
  NR_DPUS             = 8
else
  NR_COLS             ?= 16
  NR_DPUS             ?= 32
endif

# Version information
VERSION             = 0.0.0

# Application sources and artifacts
APP_BIN             = $(BUILD_DIR)/$(PROJECT)
APP_LIB             = $(BUILD_DIR)/$(PROJECT_LIB)
APP_SOURCES         =
APP_MAIN            = src/emb_host.c
APP_OBJS            = $(patsubst %.c,$(BUILD_DIR)/%.o,$(APP_SOURCES) $(APP_MAIN))

# Includes
INC                 = -Iinclude

# Test sources and artifacts
TEST_BIN            = $(BUILD_DIR)/$(PROJECT)_tests
TEST_SOURCES        = $(APP_SOURCES) tests/main.c
TEST_OBJS           = $(patsubst %.c,$(BUILD_DIR)/tests/%.o,$(TEST_SOURCES))

# Generated dependency files
DEPS                = $(APP_OBJS:.o=.d) \
                      $(TEST_OBJS:.o=.d)

# Compiler options
CC                  = gcc
COMMON_CFLAGS       = -lm --std=c11 -fPIC # -Wall -Wextra -MMD -Werror
SHARED_CFLAGS       = -shared -Wl,-soname,$(PROJECT)
DPU_OPTS            = `dpu-pkg-config --cflags --libs dpu`

# Tools
CLANG_FORMAT        = clang-format

# Debug/Release mode
ifneq ($(DEBUG),)
  COMMON_CFLAGS     += -g -DDEBUG
  BUILD_DIR         := $(BUILD_DIR)/debug
else
  COMMON_CFLAGS     += -O3
  BUILD_DIR         := $(BUILD_DIR)/release
endif

CFLAGS              += $(COMMON_CFLAGS) \
                       -DVERSION=$(VERSION) \
                       -DNR_DPUS=$(NR_DPUS) \
                       -DNR_TASKLETS=$(NR_TASKLETS) \
                       -DCOUNTER_CONFIG=$(COUNTER_CONFIG) \
                       -DNR_TABLES=$(NR_TABLES) \
                       -DNR_COLS=$(NR_COLS) \
                       -DMAX_NR_BATCHES=$(MAX_NR_BATCHES) \
                       -DHOST=1 \
                       -DDPU_BINARY=\"dpu/$(EXE_DPU)\" \
                       -DMAX_ENC_BUFFER_MB="$(MAX_ENC_BUFFER_MB)" \
                       -D_POSIX_C_SOURCE=199309L # For clock_gettime

# define SHOW_DPU_LOGS in the source if we want DPU logs
ifeq ($(SHOW_DPU_LOGS), 1)
	CFLAGS+=-DSHOW_DPU_LOGS
endif

# Silence make
ifneq ($(V),)
  SILENCE           =
else
  SILENCE           = @
endif

# Fancy output
SHOW_COMMAND        := @printf "%-15s%s\n"
SHOW_CC             := $(SHOW_COMMAND) "[ $(CC) ]"
SHOW_CLEAN          := $(SHOW_COMMAND) "[ CLEAN ]"
SHOW_GEN            := $(SHOW_COMMAND) "[ GEN ]"
SHOW_MAKE           := $(SHOW_COMMAND) "[ MAKE ]"
SHOW_FORMAT         := $(SHOW_COMMAND) "[ FORMAT ]"

##############################################################################################
# Default target and help message
##############################################################################################
DEFAULT_TARGET =  $(APP_BIN)

all: $(DEFAULT_TARGET) $(APP_LIB) dpu
.PHONY: all

# Take care of compiler generated depedencies
-include $(DEPS)

##############################################################################################
# Application
##############################################################################################
$(APP_BIN): $(APP_OBJS)
	$(SHOW_CC) $@
	$(SILENCE)$(CC) -o $@ $(APP_OBJS) $(DPU_OPTS)

$(APP_LIB): $(APP_OBJS)
	$(SHOW_CC) $@
	$(SILENCE)$(CC) -o $@ $(APP_OBJS) $(DPU_OPTS) $(SHARED_CFLAGS)

$(BUILD_DIR)/%.o: %.c
	$(SHOW_CC) $@
	$(SILENCE)mkdir -p $(dir $@)
	$(SILENCE)$(CC) $(CFLAGS) $(INC) -c $< -o $@ $(DPU_OPTS)

##############################################################################################
# DPU Application
##############################################################################################

export BUILD_DIR
export DEBUG
export NR_DPUS
export NR_TASKLETS
export COUNTER_CONFIG
export EXE_DPU
export NR_TABLES
export NR_COLS
export MAX_NR_BATCHES

dpu:
	$(SHOW_MAKE) $@
	$(SILENCE)$(MAKE) -C src/dpu

##############################################################################################
# Tests
##############################################################################################



##############################################################################################
# Cleanup
##############################################################################################
clean:
	$(SHOW_CLEAN) $(BUILD_DIR)
	$(SILENCE)rm -rf $(BUILD_DIR)

.PHONY: clean

##############################################################################################
# Format
##############################################################################################
format:
	$(SHOW_FORMAT) $@
	$(SILENCE)$(CLANG_FORMAT) -i src/*.c # src/*.h include/*.h
	$(SILENCE)$(MAKE) format -C src/dpu

.PHONY: format
