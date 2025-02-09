/**
 * @file fifo.h
 * @author Dimitri Gerin (dgerin@upmem.com)
 * @brief fifo header file (CPU side)
 * @copyright 2022 UPMEM
 */

#ifndef __FIFO__H_
#define __FIFO__H_
#include "sys/types.h"
#include <pthread.h>
#include <semaphore.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * @brief FIFO : asnchronous FIFO (FIFO for multithreded env)
 *
 * @param pthread_mutex_t main mutex
 * @param pthread_cond_t main wait conditon
 * @param uint64_t r read index
 * @param uint64_t w write index
 * @param uint64_t depth nr or items in FIFO
 * @param bool full full state
 * @param bool empty empty state
 * @param void **items items buffer
 */
typedef struct FIFO {
  pthread_mutex_t lock;
  pthread_cond_t condw, condr;
  uint64_t r;
  uint64_t w;
  uint64_t depth;
  bool full;
  bool empty;
  void **items;
} FIFO;

void FIFO_INIT(FIFO *this, void *items, uint64_t nr_items, uint64_t item_size,
               uint64_t nproducer, uint64_t nconsumer);
void FIFO_FREE(FIFO *this);

void FIFO_PUSH_RELEASE_ITEM(FIFO *this);
void FIFO_POP_RELEASE_ITEM(FIFO *this);

void *FIFO_PUSH_RESERVE_ITEM(FIFO *this);
void *FIFO_POP_RESERVE_ITEM(FIFO *this);

/**
 * @brief FIFO_PUSH_RESERVE macro wrapper for FIFO_PUSH_RESERVE_ITEM
 * @param TYPE_ item type
 * @param FIFO_ fifo pointer
 */
#define FIFO_PUSH_RESERVE(TYPE_, FIFO_)                                        \
  (TYPE_ *)(FIFO_PUSH_RESERVE_ITEM(&(FIFO_)))

/**
 * @brief FIFO_POP_RESERVE macro wrapper for FIFO_POP_RESERVE_ITEM
 * @param TYPE_ item type
 * @param FIFO_ fifo pointer
 */
#define FIFO_POP_RESERVE(TYPE_, FIFO_)                                         \
  (TYPE_ *)(FIFO_POP_RESERVE_ITEM(&(FIFO_)))

/**
 * @brief FIFO_PUSH_RELEASE macro wrapper for FIFO_PUSH_RELEASE_ITEM
 * @param FIFO_ fifo pointer
 */
#define FIFO_PUSH_RELEASE(FIFO_) FIFO_PUSH_RELEASE_ITEM(&(FIFO_))

/**
 * @brief FIFO_POP_RELEASE macro wrapper for FIFO_POP_RELEASE_ITEM
 * @param FIFO_ fifo pointer
 */
#define FIFO_POP_RELEASE(FIFO_) FIFO_POP_RELEASE_ITEM(&(FIFO_))

#endif
