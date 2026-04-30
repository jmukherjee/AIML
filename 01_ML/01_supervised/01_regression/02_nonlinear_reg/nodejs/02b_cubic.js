// ML: Non-Linear Regression - Polynomial / Cubic
// (Node.js port of ../jupyter/02b_cubic.ipynb)
//
// Neighborhood Quality Index (0-100) vs price: S-curve — flat in rough zones,
// rising through working/middle-class, surging in luxury enclaves.
//
//   price = a * nqi^3 + b * nqi^2 + c * nqi + d

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml, linspace } from './plot.js';

const data = {
  nqi:   [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100],
  price: [180, 195, 215, 250, 310, 410, 560, 760, 900, 1080, 1320, 1600],
};
const test = { nqi: [25, 55, 75, 92] };

console.table(data.nqi.map((n, i) => ({ nqi: n, price: data.price[i] })));

// Cubic features: [nqi, nqi^2, nqi^3]
const X = data.nqi.map((x) => [x, x * x, x * x * x]);
const Y = data.price.map((p) => [p]);

const reg = new MLR(X, Y);

const c1 = reg.weights[0][0];
const c2 = reg.weights[1][0];
const c3 = reg.weights[2][0];
const d  = reg.weights[3][0];
console.log(
  `fit: price = ${c3.toFixed(4)} * nqi^3 + ${c2.toFixed(4)} * nqi^2 + ${c1.toFixed(4)} * nqi + ${d.toFixed(4)}`,
);

const Xtest = test.nqi.map((x) => [x, x * x, x * x * x]);
const predictions = reg.predict(Xtest).map((row) => row[0]);
console.table(test.nqi.map((x, i) => ({ nqi: x, price: predictions[i] })));

const nqiGrid = linspace(Math.min(...data.nqi), Math.max(...data.nqi), 200);
const curve = nqiGrid.map((x) => reg.predict([[x, x * x, x * x * x]])[0][0]);

writePlotlyHtml(
  new URL('./02b_cubic_plot.html', import.meta.url).pathname,
  'Cubic Regression - NQI vs price',
  [
    { type: 'scatter', mode: 'markers', name: 'training',
      x: data.nqi, y: data.price, marker: { color: 'magenta', symbol: 'star', size: 12 } },
    { type: 'scatter', mode: 'lines', name: 'cubic fit',
      x: nqiGrid, y: curve, line: { color: 'blue' } },
    { type: 'scatter', mode: 'markers', name: 'predicted',
      x: test.nqi, y: predictions, marker: { color: 'red', size: 10 } },
  ],
  { xaxis: { title: 'Neighborhood Quality Index' }, yaxis: { title: 'price ($k)' } },
);

console.log('\nWrote 02b_cubic_plot.html');
