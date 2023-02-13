# Getting Started

```cpp
#include<librapid>

int main() {
    fmt::print("Hello, World\n");
}
```

```cpp
#include<librapid>

int main() {
    lrc::Array<float> arr(lrc::Shape({3, 3}));
    arr << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    fmt::print("arr: {}\n", arr);
}
```

All done!