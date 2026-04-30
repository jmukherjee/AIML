// ML: Non-Linear Regression - Polynomial / Quadratic
// (Node.js port of ../jupyter/02a_quadratic.ipynb)
//
// House age vs price: ∪-parabola — new builds priced high, mid-age homes dip,
// historic homes recover.
//
//   price = a * age^2 + b * age + c

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml, linspace } from './plot.js';

const data = {
  age:   [1, 5, 10, 15, 20, 25, 30, 40, 55, 70, 85, 100],
  price: [780, 690, 600, 540, 510, 500, 510, 560, 660, 770, 880, 990],
};
const test = { age: [3, 22, 45, 90] };

console.table(data.age.map((a, i) => ({ age: a, price: data.price[i] })));

// Quadratic features: [age, age^2]
const X = data.age.map((x) => [x, x * x]);
const Y = data.price.map((p) => [p]);

const reg = new MLR(X, Y);

// weights shape: 3 x 1 → [coef_age, coef_age2, intercept]
const b = reg.weights[0][0];
const a = reg.weights[1][0];
const c = reg.weights[2][0];
console.log(`fit: price = ${a.toFixed(4)} * age^2 + ${b.toFixed(4)} * age + ${c.toFixed(4)}`);

const Xtest = test.age.map((x) => [x, x * x]);
const predictions = reg.predict(Xtest).map((row) => row[0]);
console.table(test.age.map((x, i) => ({ age: x, price: predictions[i] })));

const ageGrid = linspace(Math.min(...data.age), Math.max(...data.age), 200);
const curve = ageGrid.map((x) => reg.predict([[x, x * x]])[0][0]);

writePlotlyHtml(
  new URL('./02a_quadratic_plot.html', import.meta.url).pathname,
  'Quadratic Regression - house age vs price',
  [
    { type: 'scatter', mode: 'markers', name: 'training',
      x: data.age, y: data.price, marker: { color: 'magenta', symbol: 'star', size: 12 } },
    { type: 'scatter', mode: 'lines', name: 'quadratic fit',
      x: ageGrid, y: curve, line: { color: 'blue' } },
    { type: 'scatter', mode: 'markers', name: 'predicted',
      x: test.age, y: predictions, marker: { color: 'red', size: 10 } },
  ],
  { xaxis: { title: 'house age (yr)' }, yaxis: { title: 'price ($k)' } },
);

console.log('\nWrote 02a_quadratic_plot.html');
