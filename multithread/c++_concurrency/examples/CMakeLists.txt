list(APPEND EXAMPLES
  SimpleThread
  BasicThread
  ThreadId
  ThreadMutex
  LockGuard
  ConditionVariable
  SumThread
  SumThreadTask
)

include(cmake_utility)

list(APPEND PUBLIC_LIBS
)

list(APPEND PRIVATE_LIBS
  Threads::Threads
)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
endforeach(EXAMPLE)
