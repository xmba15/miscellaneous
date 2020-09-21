# Thread Identification
- thread ID: is represented by pthread_t data type. A function must be userd to compared two thread IDs.

```c
#include <pthread.h>

/**
 * @param threadIDs
 * @return nonzero if equal, 0 otherwise
 */
int pthread_equal(pthread_t tid1, pthread_t tid2);
```

- A thread can obtain its own thread ID by calling the pthread_self function

```c
#include <pthread.h>

/**
 * @return the thread ID of the calling thread
 */
pthread_t pthread_self(void);
```

# Thread Creation
- With pthreads, when a program runs, it starts out as a single process with a single thread of control. As the program runs, its behavior should be indistinguishable from the traditional process, until it creates more threads.
- Additional threads can be created with pthread_create function

```c
#include <pthread.h>

/**
 * @return 0 if OK, error number on failure
 */
int pthread_create(pthread_t *restrict tidp,
    const pthread_attr_t *restrict attr,
    void *(*start_rtn)(void *), void *restrict arg);
```
