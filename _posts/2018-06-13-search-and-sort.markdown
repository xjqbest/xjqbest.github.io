---    
layout: post
title:  "Searching and Sorting小笔记"
date:   2018-06-13 11:00:01
categories: ALGORITHM
tags: ALGORITHM
excerpt: 
---

# Search

## Binary Search

```cpp
// arr is a sorted array in acsending order
int binary_search(vector<int>& arr, int target) {
    int l = 0;
    int r = arr.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    } 
}
```

### 搜索旋转排序数组

[题目链接](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/)

```cpp
int search(vector<int>& nums, int target) {
    int l = 0;
    int r = nums.size() - 1;
    while(l < r) {
        int mid = l + (r - l) / 2;
        //cout << mid << " " << l << " " << r << nums[mid] << "vs" <<nums[l] << endl;
        if (nums[mid] > nums[r]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    int start = 0;
    int end = nums.size() - 1;
    while (start <= end) {
        int mid = start + (end - start) / 2;
        int real_mid = (l + mid) % nums.size();
        if (nums[real_mid] == target) {
            return real_mid;
        } else if (nums[real_mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return -1;
}
```

### Search in an almost sorted array

[链接](https://www.geeksforgeeks.org/search-almost-sorted-array/)

给定一几乎排好序的数组，arr[i]可能在arr[i+1]或者arr[i-1]

```cpp
int binary_search(vector<int>& arr, int target) {
    int l = 0;
    int r = arr.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if(mid > 0 && arr[mid - 1] == target) {
            return mid - 1;
        } else if (mid < arr.size() - 1 && arr[mid + 1] == target) {
            return mid + 1;
        } else if (arr[mid] < target) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    } 
}
```

### Find k closest elements to a given value

Given a sorted array arr[] and a value X, find the k closest elements to X in arr[]. 

[链接](https://www.geeksforgeeks.org/find-k-closest-elements-given-value/)

若数组中有重复元素，可以先找到lower_bound和upper_bound，然后再计算最近的k个。

时间复杂度O(logn + k)

# Sort

## MergeSort

平均和最坏时间复杂度O(nlogn)，空间复杂度O(n)
```cpp
void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int mid = l + (r - l) / 2;
        merge_sort(arr, l, mid);
        merge_sort(arr, mid + 1, r);
        merge(arr, l , mid, r);
    }
}
```
其中merge函数是将数组的[l, mid]与[mid + 1, r]合并。

### Union and Intersection of two sorted arrays

[链接]（https://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/)

求两个已排序的数组的并集或者交集，可以借助于归并排序。

## QuickSort

跟归并排序一样，快排也是一种分治算法（Divide and Conquer）。
平均时间复杂度O(nLogn)，最坏情况下是O(n^2)。

```cpp
// [low, high]
void quickSort(arr, low, high) {
    if (low < high)
    {
        /* pi is partitioning index, arr[pi] is now at right place */
        pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);  // Before pi
        quickSort(arr, pi + 1, high); // After pi
    }
}
```
其中实现的关键是partition函数，该函数选取一个元素并将其放到排序后的正确位置上。 

选取一个元素(pivot)又有以下几种选法：
1. 取第一个
2. 取最后一个
3. 取中间
4. 随机选取一个

下面是其中一种做法：
```cpp
// [l, r]
int partition(arr, l , r) {
    int p = arr[r];
    int s = l - 1;
    for (int i = l; i <= r - 1; ++i) {
        if (arr[i] <= p) {
            swap(arr[++s], arr[i]);
        }
    }
    swap(arr[++s], arr[r]);
    return s;
}
```


### 3-way partition

如果数组有很多重复的元素，那么可以使用3-way partition，即对于数组arr[l...r]，pivot取作p：  
1. arr[0...i]的元素小于p
2. arr[j...r]的元素大于p
3. arr[i+1...j-1]的元素等于p

```cpp
// [l, r]
int partition(arr, l , r) {
    int s = l - 1;
    int e = r + 1;
    int p = arr[l];
    for (int i = l; i < e;) {
        if (arr[i] < p) {
            ++s;
            swap(arr[s], arr[i]);
            ++i;
        } else if (arr[i] > p) {
            --e;
            swap(arr[e], arr[i]);
        } else {
            ++i;
        }
    } 
}
```

### 第k个元素

```cpp
int findKth(vector<int>& nums, int k) {
        int l = 0;
        int r = nums.size() - 1;
        while(l <= r) {
            int p = partition(nums, l, r);
            if (p == k - 1) {
                return nums[p];
            } else if (p < k - 1) {
                l = p + 1;
            } else {
                r = p - 1;
            }
        }
        return INT_MAX;
}
```

### 几个问题

1. 为什么对数组倾向于快排，而对链表倾向于归并排序  
因为对于数组的归并排序，需要额外的O(n)的空间来合并数组；而对于链表的快排，随机选取一个pivot开销较大。

2. 如何迭代的实现快排，而非递归  
可以利用栈把递归算法改成迭代实现  
```cpp
void quickSortIterative (vector<int>& arr, int l, int h)
{
    stack<int> st;
    // push initial values of l and h to stack
    st.push(l);
    st.push(h);
    // Keep popping from stack while is not empty
    while (!st.empty()) {
        // Pop h and l
        h = st.top();
        st.pop();
        l = st.top();
        st.pop();
        // Set pivot element at its correct position
        // in sorted array
        int p = partition(arr, l, h);
        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            st.push(l);
            st.push(p - 1);
        }
        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            st.push(p + 1);
            st.push(h);
        }
    }
}
```

3. 根据尾递归的优化，我们可以优化快排如下：  
```cpp
void quickSort(vector<int>& arr, int low, int high)
{
    while (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        low = pi + 1;
    }
}
```

## Bucket Sort

给定[0, 1]之间均匀分布的浮点数，如何快速的排序

时间复杂度O(n)

```cpp
// Function to sort arr[] of size n using bucket sort
void bucketSort(float arr[], int n)
{
    // 1) Create n empty buckets
    vector<float> b[n];
    // 2) Put array elements in different buckets
    for (int i=0; i<n; i++)
    {
       int bi = n*arr[i]; // Index in bucket
       b[bi].push_back(arr[i]);
    }
    // 3) Sort individual buckets
    for (int i=0; i<n; i++)
       sort(b[i].begin(), b[i].end());
    // 4) Concatenate all buckets into arr[]
    int index = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < b[i].size(); j++)
          arr[index++] = b[i][j];
}
```

如果有负数，那么把正数和负数分开，分别用两个桶即可。

## Counting Sort

时间复杂度O(n+k)，空间复杂度O(n+k)

```cpp
void countSort(vector<int>& arr) {
    int k = findMaxElement(arr);
    vector<int> count(k, 0);
    // count
    for (int i : arr) {
        ++count[i];
    }
    // sum
    for(int i = 1; i < arr.size(); ++i) {
        arr[i] += arr[i - 1];
    }
    // output
    vector<int> output(arr.size());
    for (int i = 0; i < arr.size(); ++i) {
        output[count[arr[i]] - 1] = arr[i];
        --count[ar[i]]; 
    }
    // copy
    for(int i = 0; i < arr.size(); ++i) {
        arr[i] = output[i];
    }
}
```

## Radix Sort

```
Do following for each digit i where i varies from least significant digit to the most significant digit:
    Sort input array using counting sort (or any stable sort) according to the i’th digit.
```

### Sort n numbers in range from 0 to n^2 – 1 in linear time

[链接](https://www.geeksforgeeks.org/sort-n-numbers-range-0-n2-1-linear-time/)



## Heap Sort

建立一个最大堆，然后依次取出堆顶元素。

首先是建堆的过程,时间复杂度 O(nLogn)，其中heapify时间复杂度O(logn)

```cpp
// To heapify a subtree rooted with node i
// n is size of heap
void heapify(vector<int>& arr, int i, int n) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[i] < arr[l]) {
        largest = l;
    }
    if (r < n && arr[i] < arr[r]) {
        largest = r;
    }
    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, largest, n);
    }
}
void build_heap(vector<int>& arr) {
    int n = arr.size();
    for (int i = n / 2 - 1; i >= 0; --i) {
        heapify(arr, i, n);
    }
}
```

建好最大堆之后，可以开始堆排序了：

```cpp
// One by one extract an element from heap
for (int i=n-1; i>=0; i--)
{
    // Move current root to end
    swap(arr[0], arr[i]);
    // call max heapify on the reduced heap
    heapify(arr, i, 0);
}
```