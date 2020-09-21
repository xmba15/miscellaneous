// Copyright (c) 2019
// All Rights Reserved.

#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <csignal>
#include <cstdlib>

void handleSignal(int signum) {
  std::cout << "Interrupt signal (" << signum << ")"
            << "\n";
  std::exit(signum);
}

int main(int argc, char *argv[]) {
  std::signal(SIGINT, handleSignal);

  int i = 0;
  while (++i) {
    std::cout << "Going to sleep..."
              << "\n";
    if (i == 3) {
      // raise a signal
      raise(SIGINT);
    }

    sleep(1);
  }
  return 0;
}
