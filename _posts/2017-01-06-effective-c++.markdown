---
layout: post
title:  "One rule a day, keeps c++ effective"
date:   2017-01-06 23:00:00
categories: C++
tags: C++
excerpt: One rule a day, keeps c++ effective
---

## Learn one rule per day in book <<Effective C++>>.

### Rule01

视C++为一个语言联邦。

C++ 包含四个次语言：

1. C

2. Object-Oriented C++

3. Template C++

4. STL

可以理解为C++四大组成部分，对这四大部分可能编程的策略也不相同。高效编程守则视状况而变化，取决于使用的是哪一部分。

### Rule02

尽量以const，enum，inline替换#define

#### 使用const或enum替换#define

对于 #define RATIO 0.5 ，记号名称RATIO可能从未被编译器看见，因为#define是在编译器开始处理源码之前它就被处理器移走了，
所以并未进入记号表(symbol table)，于是debug的时候就看不到它。

当使用const来代替#define时候，有两个比较特殊的情况：

1. 定义常量指针时候。  
例如在头文件中声明常量字符串，需要这样写： const char* const p = "hello world"

2. class中的常量。

	const常量的初始化必须在构造函数初始化列表中初始化。
	
	如果要确保此常量只有一份实体，需将其声明为 static：
	

		class Person {
		...
		private:
			static const double height;
		...
		};
		
	const double Person::height = 176.0;

	
	如果编译器不支持"static整型常量的in-class初值设定"（现在一般是支持的），则可以使用enum，
	enum比较像#define在于取enum地址不合法。
	

		class Person {
		...
		private:
			enum { NumHobbies = 5 };
			int hobbies[NumHobbies];
		}	


#### 使用inline函数替换#define实现的宏(macro)

比如有：

```
#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))
```

我们可以使用template inline函数替换它，下面的call_with_max是真正的函数，遵循作用域和访问规则:

```
template <typename T>
inline void call_with_max(const T& a, const T& b) {
	f(a > b ? a : b);
}
```

### Rule03

尽可能使用const

const可以修饰

1. 指针、迭代器、引用 以及它们所指的对象

2. 函数返回值、函数参数、函数本身（成员函数）

3. 修饰常量、static对象，类中的static和non-static成员变量


使用const可以帮助编译器检测错误用法，例如 if (a * b = c) ...

当const和non-const成员函数有着实质等价的实现时，令non-const版本调用const版本来避免代码重复。

bitwise constness : const成员函数不更改对象任何的non-static成员变量。

### Rule04

确定对象被使用前已被初始化

1. 在使用对象前应先将它初始化，对于内置类型，需要手工的初始化。

2. 对于类中的构造函数，要确保每个构造函数把每个成员初始化。并总是使用成员初值列（member initialization list）。

	当成员变量是const或引用时，是必须使用初始列的。
	
	c++的成员初始化顺序：先初始化基类，再按声明顺序初始化成员变量。因此初值列列出的成员变量，次序应该跟声明次序一样。

3. c++对于定义于不同的编译单元的non-local static对象的初始化次序无明确定义。

	static对象：寿命为从被构造出来直到程序结束为止。包含全局对象，namespace作用域的对象，以及在类中、函数中、文件的作用域中被声明为static的对象。
	
	函数内的static对象称为local static对象，其他static对象称为non-local static对象。
	
	为了避免跨编译单元的初始化问题，可以以local static对象替换non-local static对象。（即singleton模式）

### Rule05

了解c++默认编写并调用哪些函数

只要没有显式定义复制构造函数，编译器就会自动生成一个。另一方面，只要定义了任何构造函数，编译器就不会生成默认构造函数。  
可以通过定义显式默认或者显式删除构造函数来影响自动生成的默认构造函数和默认复制构造函数。

### Rule06

若不想使用编译器自动生成的函数，就该明确拒绝

为避免编译器自动生成一些函数（默认构造函数或者默认复制构造函数），可以将相应的成员函数声明为private并且不给出实现。
或者使用基类，基类中把相应的成员函数声明为private并且不给出实现。

### Rule07

为多态基类声明virtual析构函数

如果class带有任何virtual 函数，那么它就应该有virtual析构函数。  
如果一个类的设计不是为了作为基类或者具备多态，就不应该声明virtual函数。  
如果需要一个抽象基类，一个做法是将析构函数声明为pure virtual，并为之提供一个定义。

### Rule08

别让异常逃离析构函数

析构函数绝对不要吐出异常。如果一个被析构函数调用的函数可能抛出异常，析构函数应该捕捉异常，然后吐下异常或者结束程序。  
如果用户需要对某个操作函数运行期间跑出的异常作反应，那么应该提供一个普通函数执行操作，而不是在析构函数中。

### Rule09

在构造和析构期间不要调用virtual函数，因为这类调用从不下降至derived class（比起当前执行构造函数和析构函数的那一层）。 
在derived class对象的base class构造期间，对象的类型是base class而不是derived class。

### Rule10

令operator=返回一个reference to *this

### Rule11

确保当对象自我赋值时operator=有良好的行为。（源对象和目标对象的地址的比较、语句顺序）  
确定任何函数如果操作一个以上的对象，而其中多个对象是同一个对象时，其行为依然正确。

### Rule12

Copying函数应该确保复制“对象内的所有成员变量”及“所有base class成分”。（复制所有local成员变量，调用所有base class内的适当的copying函数）  
不要尝试以某个copying函数实现另一个copying函数，应该将共同的部分放进第三个函数中，并由两个copying函数共同调用。

### Rule13

为了防止资源泄漏，使用RAII对象，它们在构造函数中获得资源并在析构函数中释放资源。

### Rule14

复制RAII对象必须一并复制它所管理的资源，所以资源的copying行为决定了RAII对象的copying行为。  
普遍而常见的RAII class copying行为是：禁止复制、引用计数法（reference counting）、复制底部资源（deep copying）、转移底部资源所有权。