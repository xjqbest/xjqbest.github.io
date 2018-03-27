---
layout: post
title:  "Lower_Bound和Upper_Bound的实现"
date:   2016-08-20 23:54:02
categories: Search
tags: BinarySearch
excerpt: Lower_Bound和Upper_Bound的实现
---

# Lower_Bound和Upper_Bound的实现

lower_bound和upper_bound是C++的标准函数,所以日常使用完全没有必要自己实现,不过,这么经典的算法,还是得要求自己能够秒秒钟搞定,编译运行无错.

lower_bound和upper_bound不完全等同于二分查找,它俩返回的不是有没有,而是返回新数据的插入位置.

lower_bound和upper_bound的差别只有一个小于号和小于等于号的差别,以下是我的实现:

```cpp
int lower_bound(int arr[], int len, int val)
{
    int beg = 0;
    int end = len;
    int mid;
    while (beg < end) {
        mid = (beg + end) >> 1;
        if (arr[mid] < val) {
            beg = mid + 1;
        } else {
            end = mid;
        }   
    }   
    return beg;
}
```

```cpp
int upper_bound(int arr[], int len, int val)
{
    int beg = 0;
    int end = len;
    int mid;
    while (beg < end) {
        mid = (beg + end) >> 1;
        if (arr[mid] <= val) {
            beg = mid + 1;
        } else {
            end = mid;
        }   
    }   
    return beg;
}
```

试用举例:

```cpp
int main()
{
    int arr[] = {1, 3, 4, 8, 8, 10, 12, 23};
    cout << lower_bound(arr, 8, 8) << endl;
    cout << lower_bound(arr, 8, 88) << endl;
    cout << lower_bound(arr, 8, 0) << endl;
    return 0;
}
```