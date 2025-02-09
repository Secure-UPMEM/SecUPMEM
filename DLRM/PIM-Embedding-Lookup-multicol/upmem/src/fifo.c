/**
 * @file fifo.c
 * @author Dimitri Gerin (dgerin@upmem.com)
 * @brief fifo functions (CPU side)
 * @copyright 2022 UPMEM
 */

#include "fifo.h"

/**
 * @brief fifo_push_update : update fifo state on write
 *
 * @param this fifo pointer
 */
void fifo_push_update(FIFO *this) {
  this->w = (this->w + 1) % this->depth;
  this->empty = false;
  if (this->w == this->r)
    this->full = true;
}

/**
 * @brief fifo_pop_update : update fifo state on read
 *
 * @param this fifo pointer
 */
void fifo_pop_update(FIFO *this) {
  this->r = (this->r + 1) % this->depth;
  this->full = false;
  if (this->w == this->r)
    this->empty = true;
}

/**
 * @brief FIFO_INIT : initialize FIFO
 *
 * @param this fifo pointer
 * @param items pointer on pre allocated buffer of items
 * @param nr_irems number of items
 * @param item_size size of item (in byte)
 * @param nproducer number of producer (PUSH)
 * @param nconsumer number of consumer (GET)
 */
void FIFO_INIT(FIFO *this, void *items, uint64_t nr_items, uint64_t item_size,
               uint64_t nproducer, uint64_t nconsumer) {
  this->r = 0;
  this->w = 0;
  this->depth = nr_items;
  this->full = false;
  this->empty = true;

  this->items = malloc(this->depth * sizeof(void *));

  for (uint64_t i = 0; i < this->depth; i++) {
    this->items[i] = items + (i * item_size);
  }

  // TODO : replace conw, condr with single variable cond
  pthread_mutex_init(&(this->lock), NULL);
  pthread_cond_init(&(this->condw), NULL);
  pthread_cond_init(&(this->condr), NULL);
}

/**
 * @brief FIFO_FREE : free FIFO
 *
 * @param this fifo pointer
 */
void FIFO_FREE(FIFO *this) { free(this->items); }

/**
 * @brief FIFO_PUSH_RESERVE_ITEM : producer side item slot reservation
 *
 * @param this fifo pointer
 */
void *FIFO_PUSH_RESERVE_ITEM(FIFO *this) {
  void *item;
  pthread_mutex_lock(&(this->lock));
  if (this->full)
    pthread_cond_wait(&(this->condw), &(this->lock));
  item = this->items[this->w];
  pthread_mutex_unlock(&(this->lock));
  return item;
}

/**
 * @brief FIFO_PUSH_RESERVE_ITEM : producer side item slot releasing (PUSH)
 *
 * @param this fifo pointer
 */
void FIFO_PUSH_RELEASE_ITEM(FIFO *this) {
  pthread_mutex_lock(&(this->lock));
  fifo_push_update(this);
  pthread_cond_signal(&(this->condr));
  pthread_mutex_unlock(&(this->lock));
}

/**
 * @brief FIFO_POP_RESERVE_ITEM : consumer side item slot reservation (GET)
 *
 * @param this fifo pointer
 */
void *FIFO_POP_RESERVE_ITEM(FIFO *this) {
  void *item;
  pthread_mutex_lock(&(this->lock));
  if (this->empty)
    pthread_cond_wait(&(this->condr), &(this->lock));
  item = this->items[this->r];
  pthread_mutex_unlock(&(this->lock));
  return item;
}

/**
 * @brief FIFO_POP_RELEASE_ITEM : consumer side item slot releasing
 *
 * @param this fifo pointer
 */
void FIFO_POP_RELEASE_ITEM(FIFO *this) {
  pthread_mutex_lock(&(this->lock));
  fifo_pop_update(this);
  pthread_cond_signal(&(this->condw));
  pthread_mutex_unlock(&(this->lock));
}