{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2A"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Function to generate an identity matrix of size n\n",
    "def identity_matrix(n):\n",
    "    a = [[0 for y in range(n)] for x in range(n)]\n",
    "    for i in range(n):\n",
    "        a[i][i] = 1\n",
    "    return a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]  is the identity matrix\n"
     ]
    }
   ],
   "source": [
    "n = 5\n",
    "A = identity_matrix(n)\n",
    "print(A,\" is the identity matrix\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2B"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1, 0, 2, 0, 0], [0, 1, 2, 0, 0], [0, 0, 2, 0, 0], [0, 0, 2, 1, 0], [0, 0, 2, 0, 1]]  after changing third column to 2\n"
     ]
    }
   ],
   "source": [
    "# Changing third column of a to 2\n",
    "for i in range(n):\n",
    "    A[i][2] = 2\n",
    "print(A, \" after changing third column to 2\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2C"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sum of all the elements in the matrix  14\n"
     ]
    }
   ],
   "source": [
    "#Determining sum of all elements in the matrix\n",
    "total_sum = 0\n",
    "for x in A:\n",
    "    total_sum += sum(x)\n",
    "\n",
    "print(\"sum of all the elements in the matrix \", total_sum)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [2, 2, 2, 2, 2], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]  is the transposed matrix\n"
     ]
    }
   ],
   "source": [
    "#Matrix transposition\n",
    "A = [[A[j][i] for j in range(n)] for i in range(n)]\n",
    "\n",
    "print(A, \" is the transposed matrix\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2E"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10\n",
      "6\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "#Determining sum of third row, sum of the diagonal and sum of the second column in the matrix\n",
    "sum_of_third_row = sum(A[2][i] for i in range(n))\n",
    "\n",
    "print(sum_of_third_row)\n",
    "\n",
    "sum_of_diag = sum(A[i][i] for i in range(n))\n",
    "\n",
    "print(sum_of_diag)\n",
    "\n",
    "sum_of_second_col = sum(A[i][1] for i in range(n))\n",
    "\n",
    "print(sum_of_second_col)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]\n",
      " [-0.23413696  1.57921282  0.76743473 -0.46947439  0.54256004]\n",
      " [-0.46341769 -0.46572975  0.24196227 -1.91328024 -1.72491783]\n",
      " [-0.56228753 -1.01283112  0.31424733 -0.90802408 -1.4123037 ]\n",
      " [ 1.46564877 -0.2257763   0.0675282  -1.42474819 -0.54438272]]\n"
     ]
    }
   ],
   "source": [
    "# Generating a standard normal matrix B. A standard normal matrix is a matrix with mean value 0 and standard deviation 1\n",
    "import numpy as np\n",
    "np.random.seed(42)\n",
    "B = np.random.standard_normal(size = (5,5))\n",
    "\n",
    "print(B)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2G"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.49671415 -1.1382643   0.64768854  1.52302986 -0.23415337]\n",
      " [ 1.46564877 -0.2257763   0.0675282  -0.42474819 -0.54438272]]  is the resultant matrix\n"
     ]
    }
   ],
   "source": [
    "# Generating matrix C\n",
    "C = np.zeros((2,5))\n",
    "C[0] = B[0] - A[1]\n",
    "C[1] = A[3] + B[4]\n",
    "print(C, \" is the resultant matrix\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2H"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[array([0.99342831, 2.93129754]), array([-3.4147929, -0.6773289]), array([2.59075415, 0.27011282]), array([ 7.61514928, -2.12374093]), array([-1.40492025, -3.26629635])]\n"
     ]
    }
   ],
   "source": [
    "D = [C[:,i] * (i+2) for i in range(n)]\n",
    "print(D)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2I"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 6.66666667 -3.33333333  6.66666667]\n",
      " [-3.33333333  1.66666667 -3.33333333]\n",
      " [ 6.66666667 -3.33333333  6.66666667]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 1., -1.],\n",
       "       [-1.,  1.]])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# X = [1; 3; 5; 7]T , Y = [4; 3; 2; 1]T , Z = [2; 4; 6; 8]T. Find covariance matrix of X,Y,Z\n",
    "X = np.array([1,3,5,7])\n",
    "Y = np.array([4,3,2,1])\n",
    "Z = np.array([2,4,6,8])\n",
    "\n",
    "print(np.cov([X,Y,Z]))\n",
    "\n",
    "# Pearson correlation co-efficient of X and Y\n",
    "np.corrcoef(X,Y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2J"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LHS Mean is :  108.71428571428571\n",
      "RHS Mean for Population Standard Deviation 108.71428571428572\n",
      "RHS Mean for Sampling Standard Deviation 114.99319727891157\n"
     ]
    }
   ],
   "source": [
    "import statistics\n",
    "data = [20, 1, 3, 5, 7, 9, 14]\n",
    "data1 = [x**2 for x in data]\n",
    "\n",
    "m1 = statistics.mean(data)\n",
    "m2 = statistics.mean(data1)\n",
    "sd1 = statistics.pstdev(data)\n",
    "sd2 = statistics.stdev(data)\n",
    "var1 = sd1 ** 2\n",
    "var2 = sd2 ** 2\n",
    "\n",
    "#LHS \n",
    "print('LHS Mean is : ', m2)\n",
    "#RHS for Population standard deviation\n",
    "print('RHS Mean for Population Standard Deviation',m1**2 + var1)\n",
    "#RHS for Sampling standard deviation\n",
    "print('RHS Mean for Sampling Standard Deviation',m1**2 + var2)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
