// ML: Non-Linear Regression - Logarithmic
// (Node.js port of ../jupyter/02c_logarithmic.ipynb)
//
// Lot size (acres) vs price: diminishing returns — first half-acre adds a lot,
// every additional acre adds less.
//
//   price = a * ln(lot_size) + b
//
// Trick: fit a straight line on (ln(x), y). Slope = a, intercept = b.

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml, linspace } from './plot.js';

const data = {
  lot_size: [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00],
  price:    [620, 690, 760, 800, 850, 890, 940, 975, 1020, 1080],
};
const test = { lot_size: [0.25, 0.80, 2.50, 4.00] };

console.table(data.lot_size.map((s, i) => ({ lot_size: s, price: data.price[i] })));

const X = data.lot_size.map((x) => [Math.log(x)]);
const Y = data.price.map((p) => [p]);

const reg = new MLR(X, Y);

const a = reg.weights[0][0];
const b = reg.weights[1][0];
console.log(`fit: price = ${a.toFixed(4)} * ln(lot_size) + ${b.toFixed(4)}`);

const Xtest = test.lot_size.map((x) => [Math.log(x)]);
const predictions = reg.predict(Xtest).map((row) => row[0]);
console.table(test.lot_size.map((x, i) => ({ lot_size: x, price: predictions[i] })));

const lotGrid = linspace(Math.min(...data.lot_size), Math.max(...data.lot_size), 200);
const curve = lotGrid.map((x) => a * Math.log(x) + b);

writePlotlyHtml(
  new URL('./02c_logarithmic_plot.html', import.meta.url).pathname,
  'Logarithmic Regression - lot size vs price',
  [
    { type: 'scatter', mode: 'markers', name: 'training',
      x: data.lot_size, y: data.price, marker: { color: 'magenta', symbol: 'star', size: 12 } },
    { type: 'scatter', mode: 'lines', name: 'log fit',
      x: lotGrid, y: curve, line: { color: 'blue' } },
    { type: 'scatter', mode: 'markers', name: 'predicted',
      x: test.lot_size, y: predictions, marker: { color: 'red', size: 10 } },
  ],
  { xaxis: { title: 'lot size (acres)' }, yaxis: { title: 'price ($k)' } },
);

console.log('\nWrote 02c_logarithmic_plot.html');
