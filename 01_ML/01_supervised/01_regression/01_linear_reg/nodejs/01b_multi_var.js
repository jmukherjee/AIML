// ML: Linear Regression - Multi Variate (Node.js port of ../jupyter/01b_multi_var.ipynb)
//
// Predict house price from house area (sq.ft.) and age (years).
//
//   y = m1*x1 + m2*x2 + b
//
// Data:
//   area  | age | price
//   2600  | 20  | 550000
//   3000  | 15  | 565000
//   3200  | 18  | 610000
//   3600  | 30  | 680000
//   4000  |  8  | 725000
//   4100  |  8  | 810000

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml } from './plot.js';

const data = {
  area:  [2600, 3000, 3200, 3600, 4000, 4100],
  age:   [20, 15, 18, 30, 8, 8],
  price: [550000, 565000, 610000, 680000, 725000, 810000],
};
const test = {
  area: [2000, 3300, 4400],
  age:  [10, 13, 16],
};

console.table(data.area.map((a, i) => ({ area: a, age: data.age[i], price: data.price[i] })));

const X = data.area.map((a, i) => [a, data.age[i]]);
const Y = data.price.map((p) => [p]);

const reg = new MLR(X, Y);

const m1 = reg.weights[0][0];
const m2 = reg.weights[1][0];
const b  = reg.weights[2][0];
console.log(`coef (area):  ${m1}`);
console.log(`coef (age):   ${m2}`);
console.log(`intercept:    ${b}`);

const Xtest = test.area.map((a, i) => [a, test.age[i]]);
const predictions = reg.predict(Xtest).map((row) => row[0]);

console.table(test.area.map((a, i) => ({ area: a, age: test.age[i], price: predictions[i] })));

// Build a regression-plane mesh over the training-data range to visualize the fit.
const areaGrid = linspace(Math.min(...data.area), Math.max(...data.area), 12);
const ageGrid  = linspace(Math.min(...data.age),  Math.max(...data.age),  12);
const z = ageGrid.map((g) => areaGrid.map((a) => reg.predict([[a, g]])[0][0]));

writePlotlyHtml(
  new URL('./01b_multi_var_plot.html', import.meta.url).pathname,
  'Linear Regression - Multi Variate',
  [
    {
      type: 'scatter3d',
      mode: 'markers',
      name: 'training',
      x: data.area,
      y: data.age,
      z: data.price,
      marker: { color: 'magenta', symbol: 'diamond', size: 6 },
    },
    {
      type: 'scatter3d',
      mode: 'markers',
      name: 'predicted',
      x: test.area,
      y: test.age,
      z: predictions,
      marker: { color: 'red', size: 6 },
    },
    {
      type: 'surface',
      name: 'fit plane',
      x: areaGrid,
      y: ageGrid,
      z,
      opacity: 0.5,
      showscale: false,
      colorscale: 'Blues',
    },
  ],
  {
    scene: {
      xaxis: { title: 'area' },
      yaxis: { title: 'age' },
      zaxis: { title: 'price' },
      camera: { eye: { x: 1.6, y: 1.6, z: 0.6 } },
    },
  },
);

console.log('\nWrote 01b_multi_var_plot.html — open it in a browser.');

function linspace(start, stop, n) {
  if (n < 2) return [start];
  const step = (stop - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + step * i);
}
