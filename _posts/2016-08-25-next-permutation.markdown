---
layout: post
title:  "Next Permuation的实现"
date:   2016-08-25 23:54:02
categories: Array
tags: Array
excerpt: Next Permuation的实现
---


# Next Permuation的实现

```cpp
bool next_permutation()
{
    for (int i = n - 1; i > 0; i--) {
        if (a[i - 1] < a[i]) {
            int j = n - 1;
            while (a[i  -1] >= a[j]) {
                j--;
            }
            swap(a[i - 1], a[j]);
            reverse(a + i, a + n);
            return true;
        }
    }
    return false;
}
```

