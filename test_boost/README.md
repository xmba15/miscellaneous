# Test Boost #
## Test Boost Container ##
### boost::container::flat_multimap ###
- Program in TestContainer.cpp compares the insertion execution time of boost's flat\_multimap (implemented with sorted arrays) and stl's multimap (implemented with a self-balanced bst).
- result:

```
boost flat multimap: 10848[milisec]
stl multimap: 53[milisec]
```
