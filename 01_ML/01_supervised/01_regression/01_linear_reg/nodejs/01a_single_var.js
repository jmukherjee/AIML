// ML: Linear Regression - Single Variate (Node.js port of ../jupyter/01a_single_var.ipynb)
//
// Predict house price from house area (sq.ft.).
//
//   y = m*x + b
//
// Data:
//   area  | price
//   2600  | 550000
//   3000  | 565000
//   3200  | 610000
//   3600  | 680000
//   4000  | 725000

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml } from './plot.js';

const data = {
  area:  [2600, 3000, 3200, 3600, 4000],
  price: [550000, 565000, 610000, 680000, 725000],
};
const test = {
  area: [2000, 3300, 4100],
};

console.table(data.area.map((a, i) => ({ area: a, price: data.price[i] })));

// MLR expects 2D arrays for both X and Y (one row per sample).
const X = data.area.map((a) => [a]);
const Y = data.price.map((p) => [p]);

const reg = new MLR(X, Y);

// weights shape: (nFeatures + 1) x nOutputs. Last row is the intercept.
const m = reg.weights[0][0];
const b = reg.weights[reg.weights.length - 1][0];
console.log(`coef (m):     ${m}`);
console.log(`intercept (b): ${b}`);

const Xtest = test.area.map((a) => [a]);
const predictions = reg.predict(Xtest).map((row) => row[0]);

console.table(test.area.map((a, i) => ({ area: a, price: predictions[i] })));

writePlotlyHtml(
  new URL('./01a_single_var_plot.html', import.meta.url).pathname,
  'Linear Regression - Single Variate',
  [
    {
      type: 'scatter',
      mode: 'markers',
      name: 'training',
      x: data.area,
      y: data.price,
      marker: { color: 'magenta', symbol: 'star', size: 12 },
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'predicted',
      x: test.area,
      y: predictions,
      line: { color: 'blue' },
    },
  ],
  { xaxis: { title: 'area' }, yaxis: { title: 'price' } },
);

console.log('\nWrote 01a_single_var_plot.html — open it in a browser.');
