// UI Elements
const chapterButtons = document.querySelectorAll('.chapter-btn');
const chapterContents = document.querySelectorAll('.chapter-content');
const methodSelect = document.getElementById('method-select');
const calculateBtn = document.getElementById('calculate-btn');
const resultOutput = document.getElementById('result-output');
const stepsOutput = document.getElementById('steps-output');

// Chapter navigation
chapterButtons.forEach(button => {
    button.addEventListener('click', () => {
        const chapterId = button.dataset.chapter;
        chapterContents.forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`chapter${chapterId}`).classList.add('active');
    });
});

// Numerical Methods Implementation
class NumericalMethods {
    static evaluateFunction(f, x) {
        try {
            const scope = { x };
            return math.evaluate(f, scope);
        } catch (error) {
            throw new Error('Invalid function expression');
        }
    }

    static bisection(f, a, b, tolerance, maxIterations) {
        let steps = [];
        let c;
        
        if (this.evaluateFunction(f, a) * this.evaluateFunction(f, b) >= 0) {
            throw new Error('Function must have opposite signs at endpoints');
        }

        for (let i = 0; i < maxIterations; i++) {
            c = (a + b) / 2;
            const fa = this.evaluateFunction(f, a);
            const fc = this.evaluateFunction(f, c);
            
            steps.push(`Iteration ${i + 1}: a = ${a.toFixed(6)}, b = ${b.toFixed(6)}, c = ${c.toFixed(6)}, f(c) = ${fc.toFixed(6)}`);

            if (Math.abs(fc) < tolerance) {
                return { root: c, steps };
            }

            if (fa * fc < 0) {
                b = c;
            } else {
                a = c;
            }
        }

        return { root: c, steps };
    }

    static falsePosition(f, a, b, tolerance, maxIterations) {
        let steps = [];
        let c;
        
        for (let i = 0; i < maxIterations; i++) {
            const fa = this.evaluateFunction(f, a);
            const fb = this.evaluateFunction(f, b);
            c = (a * fb - b * fa) / (fb - fa);
            const fc = this.evaluateFunction(f, c);
            
            steps.push(`Iteration ${i + 1}: a = ${a.toFixed(6)}, b = ${b.toFixed(6)}, c = ${c.toFixed(6)}, f(c) = ${fc.toFixed(6)}`);

            if (Math.abs(fc) < tolerance) {
                return { root: c, steps };
            }

            if (fa * fc < 0) {
                b = c;
            } else {
                a = c;
            }
        }

        return { root: c, steps };
    }

    static fixedPoint(f, x0, tolerance, maxIterations) {
        let steps = [];
        let x = x0;
        
        for (let i = 0; i < maxIterations; i++) {
            const xNew = this.evaluateFunction(f, x);
            const error = Math.abs(xNew - x);
            
            steps.push(`Iteration ${i + 1}: x = ${x.toFixed(6)}, f(x) = ${xNew.toFixed(6)}, error = ${error.toFixed(6)}`);

            if (error < tolerance) {
                return { root: xNew, steps };
            }

            x = xNew;
        }

        return { root: x, steps };
    }

    static newtonRaphson(f, x0, tolerance, maxIterations) {
        let steps = [];
        let x = x0;
        
        // Numerical derivative
        const h = 0.0001;
        const df = (x) => (this.evaluateFunction(f, x + h) - this.evaluateFunction(f, x - h)) / (2 * h);
        
        for (let i = 0; i < maxIterations; i++) {
            const fx = this.evaluateFunction(f, x);
            const dfx = df(x);
            
            if (Math.abs(dfx) < 1e-10) {
                throw new Error('Derivative is zero');
            }

            const xNew = x - fx / dfx;
            const error = Math.abs(xNew - x);
            
            steps.push(`Iteration ${i + 1}: x = ${x.toFixed(6)}, f(x) = ${fx.toFixed(6)}, f'(x) = ${dfx.toFixed(6)}, error = ${error.toFixed(6)}`);

            if (error < tolerance) {
                return { root: xNew, steps };
            }

            x = xNew;
        }

        return { root: x, steps };
    }

    static secant(f, x0, x1, tolerance, maxIterations) {
        let steps = [];
        let x = x1;
        let xPrev = x0;
        
        for (let i = 0; i < maxIterations; i++) {
            const fx = this.evaluateFunction(f, x);
            const fxPrev = this.evaluateFunction(f, xPrev);
            
            if (Math.abs(fx - fxPrev) < 1e-10) {
                throw new Error('Function values are too close');
            }

            const xNew = x - fx * (x - xPrev) / (fx - fxPrev);
            const error = Math.abs(xNew - x);
            
            steps.push(`Iteration ${i + 1}: x = ${x.toFixed(6)}, f(x) = ${fx.toFixed(6)}, error = ${error.toFixed(6)}`);

            if (error < tolerance) {
                return { root: xNew, steps };
            }

            xPrev = x;
            x = xNew;
        }

        return { root: x, steps };
    }
}

// Event Listeners
calculateBtn.addEventListener('click', () => {
    try {
        const method = methodSelect.value;
        const f = document.getElementById('function-input').value;
        const x0 = parseFloat(document.getElementById('x0-input').value);
        const x1 = parseFloat(document.getElementById('x1-input').value);
        const tolerance = parseFloat(document.getElementById('tolerance-input').value);
        const maxIterations = parseInt(document.getElementById('max-iterations-input').value);

        let result;
        switch (method) {
            case 'bisection':
                result = NumericalMethods.bisection(f, x0, x1, tolerance, maxIterations);
                break;
            case 'false-position':
                result = NumericalMethods.falsePosition(f, x0, x1, tolerance, maxIterations);
                break;
            case 'fixed-point':
                result = NumericalMethods.fixedPoint(f, x0, tolerance, maxIterations);
                break;
            case 'newton':
                result = NumericalMethods.newtonRaphson(f, x0, tolerance, maxIterations);
                break;
            case 'secant':
                result = NumericalMethods.secant(f, x0, x1, tolerance, maxIterations);
                break;
        }

        resultOutput.textContent = `Root: ${result.root.toFixed(6)}`;
        stepsOutput.textContent = result.steps.join('\n');
    } catch (error) {
        resultOutput.textContent = `Error: ${error.message}`;
        stepsOutput.textContent = '';
    }
});

// Linear Algebra Methods Implementation
class LinearAlgebra {
    static generateMatrixInputs(size) {
        const coefficientsDiv = document.getElementById('coefficients-input');
        const constantsDiv = document.getElementById('constants-input');
        
        // Clear existing inputs
        coefficientsDiv.innerHTML = '';
        constantsDiv.innerHTML = '';
        
        // Create coefficients matrix
        const coeffMatrix = document.createElement('div');
        coeffMatrix.className = 'matrix-input';
        
        for (let i = 0; i < size; i++) {
            const row = document.createElement('div');
            row.className = 'matrix-row';
            
            for (let j = 0; j < size; j++) {
                const cell = document.createElement('div');
                cell.className = 'matrix-cell';
                cell.innerHTML = `<input type="number" step="any" id="a${i}${j}" placeholder="0">`;
                row.appendChild(cell);
            }
            
            coeffMatrix.appendChild(row);
        }
        
        // Create constants vector
        const constVector = document.createElement('div');
        constVector.className = 'matrix-input';
        
        for (let i = 0; i < size; i++) {
            const row = document.createElement('div');
            row.className = 'matrix-row';
            
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.innerHTML = `<input type="number" step="any" id="b${i}" placeholder="0">`;
            row.appendChild(cell);
            
            constVector.appendChild(row);
        }
        
        coefficientsDiv.appendChild(coeffMatrix);
        constantsDiv.appendChild(constVector);
    }

    static getMatrixValues(size) {
        const A = [];
        const b = [];
        
        for (let i = 0; i < size; i++) {
            A[i] = [];
            for (let j = 0; j < size; j++) {
                A[i][j] = parseFloat(document.getElementById(`a${i}${j}`).value) || 0;
            }
            b[i] = parseFloat(document.getElementById(`b${i}`).value) || 0;
        }
        
        return { A, b };
    }

    static gaussianElimination(A, b, usePivoting = false) {
        const n = A.length;
        const steps = [];
        const augmented = A.map((row, i) => [...row, b[i]]);
        
        steps.push('Initial augmented matrix:');
        steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        
        // Forward elimination
        for (let i = 0; i < n; i++) {
            steps.push(`\nStep ${i + 1}: Forward Elimination`);
            
            if (usePivoting) {
                // Partial pivoting
                let maxRow = i;
                let maxVal = Math.abs(augmented[i][i]);
                
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(augmented[j][i]) > maxVal) {
                        maxRow = j;
                        maxVal = Math.abs(augmented[j][i]);
                    }
                }
                
                if (maxRow !== i) {
                    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                    steps.push(`Swapped rows ${i + 1} and ${maxRow + 1}`);
                    steps.push('Matrix after row swap:');
                    steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
                }
            }
            
            // Elimination
            for (let j = i + 1; j < n; j++) {
                const factor = augmented[j][i] / augmented[i][i];
                steps.push(`\nEliminating x${i + 1} from row ${j + 1}`);
                steps.push(`Using factor: ${factor.toFixed(6)}`);
                
                for (let k = i; k <= n; k++) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
                
                steps.push('Matrix after elimination:');
                steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
            }
        }
        
        // Back substitution
        const x = new Array(n);
        steps.push('\nBack Substitution:');
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += augmented[i][j] * x[j];
            }
            x[i] = (augmented[i][n] - sum) / augmented[i][i];
            steps.push(`x${i + 1} = (${augmented[i][n].toFixed(6)} - ${sum.toFixed(6)}) / ${augmented[i][i].toFixed(6)} = ${x[i].toFixed(6)}`);
        }
        
        return { solution: x, steps };
    }

    static gaussJordan(A, b, usePivoting = false) {
        const n = A.length;
        const steps = [];
        const augmented = A.map((row, i) => [...row, b[i]]);
        
        steps.push('Initial augmented matrix:');
        steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        
        // Forward elimination
        for (let i = 0; i < n; i++) {
            steps.push(`\nStep ${i + 1}: Forward Elimination`);
            
            if (usePivoting) {
                // Partial pivoting
                let maxRow = i;
                let maxVal = Math.abs(augmented[i][i]);
                
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(augmented[j][i]) > maxVal) {
                        maxRow = j;
                        maxVal = Math.abs(augmented[j][i]);
                    }
                }
                
                if (maxRow !== i) {
                    [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                    steps.push(`Swapped rows ${i + 1} and ${maxRow + 1}`);
                    steps.push('Matrix after row swap:');
                    steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
                }
            }
            
            // Make diagonal element 1
            const pivot = augmented[i][i];
            for (let j = i; j <= n; j++) {
                augmented[i][j] /= pivot;
            }
            steps.push(`\nMaking pivot element 1 (row ${i + 1})`);
            steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
            
            // Eliminate other rows
            for (let j = 0; j < n; j++) {
                if (j !== i) {
                    const factor = augmented[j][i];
                    for (let k = i; k <= n; k++) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                    steps.push(`\nEliminating x${i + 1} from row ${j + 1}`);
                    steps.push(augmented.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
                }
            }
        }
        
        const x = augmented.map(row => row[n]);
        steps.push('\nFinal solution:');
        x.forEach((xi, i) => steps.push(`x${i + 1} = ${xi.toFixed(6)}`));
        
        return { solution: x, steps };
    }

    static luDecomposition(A, b, usePivoting = false) {
        const n = A.length;
        const steps = [];
        const L = Array(n).fill().map(() => Array(n).fill(0));
        const U = Array(n).fill().map(() => Array(n).fill(0));
        const P = Array(n).fill().map((_, i) => i); // Permutation matrix
        
        steps.push('Starting LU Decomposition:');
        steps.push('Original matrix A:');
        steps.push(A.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        
        for (let i = 0; i < n; i++) {
            if (usePivoting) {
                // Partial pivoting
                let maxRow = i;
                let maxVal = Math.abs(A[P[i]][i]);
                
                for (let j = i + 1; j < n; j++) {
                    if (Math.abs(A[P[j]][i]) > maxVal) {
                        maxRow = j;
                        maxVal = Math.abs(A[P[j]][i]);
                    }
                }
                
                if (maxRow !== i) {
                    [P[i], P[maxRow]] = [P[maxRow], P[i]];
                    steps.push(`Swapped rows ${i + 1} and ${maxRow + 1}`);
                }
            }
            
            // Upper triangular
            for (let k = i; k < n; k++) {
                let sum = 0;
                for (let j = 0; j < i; j++) {
                    sum += L[P[i]][j] * U[j][k];
                }
                U[i][k] = A[P[i]][k] - sum;
                steps.push(`U[${i+1}][${k+1}] = ${A[P[i]][k].toFixed(6)} - ${sum.toFixed(6)} = ${U[i][k].toFixed(6)}`);
            }
            
            // Lower triangular
            for (let k = i; k < n; k++) {
                if (i === k) {
                    L[P[i]][i] = 1;
                    steps.push(`L[${i+1}][${i+1}] = 1 (diagonal element)`);
                } else {
                    let sum = 0;
                    for (let j = 0; j < i; j++) {
                        sum += L[P[k]][j] * U[j][i];
                    }
                    L[P[k]][i] = (A[P[k]][i] - sum) / U[i][i];
                    steps.push(`L[${k+1}][${i+1}] = (${A[P[k]][i].toFixed(6)} - ${sum.toFixed(6)}) / ${U[i][i].toFixed(6)} = ${L[P[k]][i].toFixed(6)}`);
                }
            }
        }
        
        steps.push('\nFinal L matrix:');
        steps.push(L.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        steps.push('\nFinal U matrix:');
        steps.push(U.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        
        // Solve Ly = Pb
        const y = new Array(n);
        steps.push('\nSolving Ly = Pb:');
        const Pb = P.map(i => b[i]);
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < i; j++) {
                sum += L[P[i]][j] * y[j];
            }
            y[i] = Pb[i] - sum;
            steps.push(`y${i + 1} = ${Pb[i].toFixed(6)} - ${sum.toFixed(6)} = ${y[i].toFixed(6)}`);
        }
        
        // Solve Ux = y
        const x = new Array(n);
        steps.push('\nSolving Ux = y:');
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += U[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / U[i][i];
            steps.push(`x${i + 1} = (${y[i].toFixed(6)} - ${sum.toFixed(6)}) / ${U[i][i].toFixed(6)} = ${x[i].toFixed(6)}`);
        }
        
        return { solution: x, steps };
    }

    static cramersRule(A, b) {
        const n = A.length;
        const steps = [];
        
        steps.push('Original matrix A:');
        steps.push(A.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
        
        // Calculate determinant with improved numerical stability
        const detA = this.determinant(A);
        steps.push(`\nStep 1: Calculate det(A) = ${detA.toFixed(6)}`);
        
        if (Math.abs(detA) < 1e-10) {
            throw new Error('Matrix is singular (determinant is zero)');
        }
        
        const x = new Array(n);
        for (let i = 0; i < n; i++) {
            steps.push(`\nStep ${i + 2}: Calculate x${i + 1}`);
            
            // Create modified matrix with improved numerical stability
            const Ai = A.map((row, j) => {
                const newRow = [...row];
                newRow[i] = b[j];
                return newRow;
            });
            
            steps.push(`Matrix A${i + 1}:`);
            steps.push(Ai.map(row => row.map(x => x.toFixed(6)).join(' ')).join('\n'));
            
            const detAi = this.determinant(Ai);
            steps.push(`det(A${i + 1}) = ${detAi.toFixed(6)}`);
            
            x[i] = detAi / detA;
            steps.push(`x${i + 1} = det(A${i + 1}) / det(A) = ${detAi.toFixed(6)} / ${detA.toFixed(6)} = ${x[i].toFixed(6)}`);
        }
        
        return { solution: x, steps };
    }

    static determinant(matrix) {
        const n = matrix.length;
        if (n === 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        
        // Use LU decomposition for better numerical stability
        const L = Array(n).fill().map(() => Array(n).fill(0));
        const U = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            // Upper triangular
            for (let k = i; k < n; k++) {
                let sum = 0;
                for (let j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = matrix[i][k] - sum;
            }
            
            // Lower triangular
            for (let k = i; k < n; k++) {
                if (i === k) {
                    L[i][i] = 1;
                } else {
                    let sum = 0;
                    for (let j = 0; j < i; j++) {
                        sum += L[k][j] * U[j][i];
                    }
                    L[k][i] = (matrix[k][i] - sum) / U[i][i];
                }
            }
        }
        
        // Determinant is product of diagonal elements of U
        let det = 1;
        for (let i = 0; i < n; i++) {
            det *= U[i][i];
        }
        
        return det;
    }
}

// Event Listeners for Linear Algebra
document.getElementById('generate-matrix-btn').addEventListener('click', () => {
    const size = parseInt(document.getElementById('matrix-size').value);
    LinearAlgebra.generateMatrixInputs(size);
});

document.getElementById('solve-linear-btn').addEventListener('click', () => {
    try {
        const method = document.getElementById('linear-method-select').value;
        const size = parseInt(document.getElementById('matrix-size').value);
        const { A, b } = LinearAlgebra.getMatrixValues(size);
        
        let result;
        switch (method) {
            case 'gauss':
                result = LinearAlgebra.gaussianElimination(A, b, false);
                break;
            case 'gauss-pivot':
                result = LinearAlgebra.gaussianElimination(A, b, true);
                break;
            case 'gauss-jordan':
                result = LinearAlgebra.gaussJordan(A, b, false);
                break;
            case 'gauss-jordan-pivot':
                result = LinearAlgebra.gaussJordan(A, b, true);
                break;
            case 'lu':
                result = LinearAlgebra.luDecomposition(A, b, false);
                break;
            case 'lu-pivot':
                result = LinearAlgebra.luDecomposition(A, b, true);
                break;
            case 'cramer':
                result = LinearAlgebra.cramersRule(A, b);
                break;
        }
        
        const resultOutput = document.getElementById('linear-result-output');
        const stepsOutput = document.getElementById('linear-steps-output');
        
        resultOutput.textContent = `Solution: [${result.solution.map(x => x.toFixed(6)).join(', ')}]`;
        stepsOutput.textContent = result.steps.join('\n');
    } catch (error) {
        document.getElementById('linear-result-output').textContent = `Error: ${error.message}`;
        document.getElementById('linear-steps-output').textContent = '';
    }
});

// Initialize matrix inputs
LinearAlgebra.generateMatrixInputs(3);

// Optimization Methods Implementation
class Optimization {
    static goldenRatio = (1 + Math.sqrt(5)) / 2;

    static goldenSectionSearch(f, a, b, tolerance, maxIterations) {
        let steps = [];
        let c = b - (b - a) / this.goldenRatio;
        let d = a + (b - a) / this.goldenRatio;
        
        for (let i = 0; i < maxIterations; i++) {
            const fc = NumericalMethods.evaluateFunction(f, c);
            const fd = NumericalMethods.evaluateFunction(f, d);
            
            steps.push(`Iteration ${i + 1}:`);
            steps.push(`a = ${a.toFixed(6)}, b = ${b.toFixed(6)}`);
            steps.push(`c = ${c.toFixed(6)}, d = ${d.toFixed(6)}`);
            steps.push(`f(c) = ${fc.toFixed(6)}, f(d) = ${fd.toFixed(6)}`);

            if (Math.abs(b - a) < tolerance) {
                const x = (a + b) / 2;
                const fx = NumericalMethods.evaluateFunction(f, x);
                return { 
                    minimum: x,
                    value: fx,
                    steps 
                };
            }

            if (fc < fd) {
                b = d;
                d = c;
                c = b - (b - a) / this.goldenRatio;
            } else {
                a = c;
                c = d;
                d = a + (b - a) / this.goldenRatio;
            }
        }

        const x = (a + b) / 2;
        const fx = NumericalMethods.evaluateFunction(f, x);
        return { 
            minimum: x,
            value: fx,
            steps 
        };
    }

    static gradientDescent(f, x0, learningRate, tolerance, maxIterations) {
        let steps = [];
        let x = x0;
        let h = 0.0001; // For numerical derivative
        
        for (let i = 0; i < maxIterations; i++) {
            // Numerical derivative
            const fx = NumericalMethods.evaluateFunction(f, x);
            const fxh = NumericalMethods.evaluateFunction(f, x + h);
            const gradient = (fxh - fx) / h;
            
            steps.push(`Iteration ${i + 1}:`);
            steps.push(`x = ${x.toFixed(6)}`);
            steps.push(`f(x) = ${fx.toFixed(6)}`);
            steps.push(`f'(x) = ${gradient.toFixed(6)}`);

            if (Math.abs(gradient) < tolerance) {
                return { 
                    minimum: x,
                    value: fx,
                    steps 
                };
            }

            x = x - learningRate * gradient;
        }

        const fx = NumericalMethods.evaluateFunction(f, x);
        return { 
            minimum: x,
            value: fx,
            steps 
        };
    }
}

// Event Listeners for Optimization
document.getElementById('optimize-btn').addEventListener('click', () => {
    try {
        const method = document.getElementById('optimization-method-select').value;
        const f = document.getElementById('optimization-function-input').value;
        const x0 = parseFloat(document.getElementById('optimization-x0-input').value);
        const learningRate = parseFloat(document.getElementById('optimization-learning-rate-input').value);
        const tolerance = parseFloat(document.getElementById('optimization-tolerance-input').value);
        const maxIterations = parseInt(document.getElementById('optimization-max-iterations-input').value);

        let result;
        switch (method) {
            case 'golden-section':
                result = Optimization.goldenSectionSearch(f, x0, x0, tolerance, maxIterations);
                break;
            case 'gradient-descent':
                result = Optimization.gradientDescent(f, x0, learningRate, tolerance, maxIterations);
                break;
        }

        const resultOutput = document.getElementById('optimization-result-output');
        const stepsOutput = document.getElementById('optimization-steps-output');
        
        resultOutput.textContent = `Minimum: ${result.minimum.toFixed(6)}, f(minimum) = ${result.value.toFixed(6)}`;
        stepsOutput.textContent = result.steps.join('\n');
    } catch (error) {
        resultOutput.textContent = `Error: ${error.message}`;
        stepsOutput.textContent = '';
    }
});

// Initialize optimization inputs
Optimization.generateOptimizationInputs(3);