# Numerical Methods Calculator

A web-based calculator for solving various numerical methods problems, including equation solving, linear algebra, and optimization.

## Features

### Chapter 1: Solving Equations
- Bisection Method
- False Position Method
- Simple Fixed Point Method
- Newton-Raphson Method
- Secant Method

### Chapter 2: Linear Algebraic Equations (Coming Soon)
- Gauss Elimination
- LU Decomposition
- Cramer's Rule
- Partial Pivoting
- Gauss-Jordan

### Chapter 3: Unconstrained Optimization (Coming Soon)
- Golden-Section Search
- Gradient Methods
- Gradients and Hessians

## Usage

1. Open `index.html` in a web browser
2. Select the desired chapter from the navigation buttons
3. Choose the numerical method you want to use
4. Enter the required parameters:
   - Function expression (e.g., `x^3 - x - 2`)
   - Initial guesses/points
   - Tolerance (default: 0.0001)
   - Maximum iterations (default: 100)
5. Click "Calculate" to see the results and step-by-step solution

## Function Input Format

The calculator uses math.js for function parsing. You can use the following operators and functions:

- Basic operators: `+`, `-`, `*`, `/`, `^` (power)
- Functions: `sin`, `cos`, `tan`, `log`, `exp`, `sqrt`, etc.
- Constants: `pi`, `e`

Examples:
- `x^2 - 4`
- `sin(x) + cos(x)`
- `exp(x) - 2`
- `log(x) - 1`

## Technical Details

- Built with HTML, CSS, and JavaScript
- Uses math.js for mathematical expression parsing
- Responsive design that works on desktop and mobile devices
- Step-by-step solution display for better understanding

## Future Improvements

- Add visualization of the functions and iterations
- Implement remaining chapters (Linear Algebra and Optimization)
- Add support for multiple variables
- Include more numerical methods
- Add export functionality for results 