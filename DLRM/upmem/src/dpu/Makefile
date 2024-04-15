CC           = dpu-upmem-dpurte-clang
CFLAGS       = --std=c11 -O2 -flto -g -Wall

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif
# define TEST in the source if we are debugging
ifeq ($(TEST), 1)
	CFLAGS+=-DTEST
endif

# Application sources and artifacts
APP_MAIN            = emb_dpu_lookup.c

EXE_DPU             ?= emb_dpu_lookup
APP_BIN             = $(BUILD_DIR)/$(EXE_DPU)
APP_SOURCES         =
APP_OBJS            = $(patsubst %.c,$(BUILD_DIR)/%.o,$(APP_SOURCES) $(APP_MAIN))
BUILD_DIR           := ../../$(BUILD_DIR)

# Includes
INC                 = -I../../include

# Generated dependency files
DEPS                = $(APP_OBJS:.o=.d) \
                      $(TEST_OBJS:.o=.d)

# Compiler options
CC                  = dpu-upmem-dpurte-clang
COMMON_CFLAGS       = --std=c11 -flto -g -Wall -Wextra -MMD -Werror

# Tools
CLANG_FORMAT        = clang-format

# Debug/Release mode
ifneq ($(DEBUG),)
  COMMON_CFLAGS     += -g -DDEBUG
endif

# define SHOW_DPU_LOGS in the source if we want DPU logs
ifeq ($(SHOW_DPU_LOGS), 1)
	COMMON_CFLAGS+=-DSHOW_DPU_LOGS
endif

BUILD_DIR           := $(BUILD_DIR)/dpu

CFLAGS              += $(COMMON_CFLAGS) \
                       -DNR_DPUS=$(NR_DPUS) \
                       -DNR_TASKLETS=$(NR_TASKLETS) \
                       -DCOUNTER_CONFIG=$(COUNTER_CONFIG) \
                       -DNR_TABLES=$(NR_TABLES) \
                       -DNR_COLS=$(NR_COLS) \
                       -DMAX_NR_BATCHES=$(MAX_NR_BATCHES)


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
SHOW_FORMAT         := $(SHOW_COMMAND) "[ FORMAT ]"

##############################################################################################
# Default target and help message
##############################################################################################
DEFAULT_TARGET =  $(APP_BIN)

all: $(DEFAULT_TARGET)
.PHONY: all

# Take care of compiler generated depedencies
-include $(DEPS)

##############################################################################################
# Application
##############################################################################################
$(DEFAULT_TARGET): $(APP_OBJS)
	$(SHOW_CC) $@
	$(SILENCE)$(CC) $(CFLAGS) -o $@ $(APP_OBJS) $(DPU_OPTS)

$(BUILD_DIR)/%.o: %.c
	$(SHOW_CC) $@
	$(SILENCE)mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ $(DPU_OPTS)

##############################################################################################
# Format
##############################################################################################
format:
	$(SHOW_FORMAT) $@
	$(SILENCE)$(CLANG_FORMAT) -i *.c # *.h

.PHONY: format
