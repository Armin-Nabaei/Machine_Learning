# Improving Performance of First-order Optimizers Based on the Scaled Gradient Technique for the First Conv Layer to Combat the Vanishing Gradient Issue.

## Algorithm

<img width="351" alt="Screenshot 2023-12-18 at 2 55 38 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/8c38d0a6-de26-4365-8da5-7d839e793e95">

## Results
<img width="1067" alt="Screenshot 2023-12-18 at 2 50 10 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/5fb6a111-55eb-4dcb-b3d1-241632cfc9e4">
__________________________________________________________________
<img width="1059" alt="Screenshot 2023-12-18 at 2 50 20 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/22747d78-61f6-4f0b-a9ad-9c8c36cae6d4">
__________________________________________________________________

<img width="760" alt="Screenshot 2023-12-18 at 2 53 32 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/b2fb6cf9-f007-403e-9cab-350b85b57ed7">
__________________________________________________________________

<img width="760" alt="Screenshot 2023-12-18 at 2 53 56 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/b8aa0268-1d34-4c48-acbc-43b491848a33">
__________________________________________________________________

#Expression Puzzle Problem

##Introduction 
Implement a dynamic programming algorithm that solves optimally the expression puzzle problem. Given a set of digits 𝑆 = {𝑆𝑖 }𝑖=1..𝑘 and an integer 𝑁, the expression puzzle problem is finding a string consisting of characters from 𝑆 (you can repeat a character as many times as you need) and the special symbols “+” and “*” (also you can repeat them as many times as you need) such that an arithmetic evaluation of the resulting expression yields the number N. For example, if 𝑆 = {2, 9}, N=229 can be obtained by creating a string by concatenating the digits 2, 2 again followed by 9: ”229”. N= 11 can be obtained by the string “2+9”, N =49 can be obtain using the string “9+2*9+22”, etc. An optimal solution is a solution that has minimal character length (i.e. any other solution string has more or equal number of characters). For example, for N=22, both “2*9+2+2” and “22” are valid puzzle solution, but only the latter is optimal. Note that for some puzzles many optimal solutions may exist, and some other puzzles may have no valid solution. 


##Specifications
 The input is specified in a file whose name is the first argument of the program. The first line contains an integer M specifying how many datasets are in the file. The reminder of the file encodes the datasets. Each dataset is encoded in one line. It starts with an integer K that indicates how many elements are in the set S, followed by the actual digits in S (you can assume that the digits do not repeat). The last number in the line is the integer number N. Note that K, {𝑆𝑖 }𝑖=1..𝑘 and N are separated by spaces. N<20000. 


##Here is an example: 
6 
2 2 9 229 
2 2 9 11 
2 2 9 729 
2 2 9 49 
3 1 4 7 21 
2 4 7 6

