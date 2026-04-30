// ML: Non-Linear Regression - Exponential
// (Node.js port of ../jupyter/02d_exponential.ipynb)
//
// Years held vs property value: compound appreciation at ~6 %/yr.
//
//   value = a * exp(r * years)
//
// Trick: fit a straight line on (x, ln(y)). Recover a = exp(intercept), r = slope.

import MLR from 'ml-regression-multivariate-linear';
import { writePlotlyHtml, linspace } from './plot.js';

const data = {
  years: [0, 2, 5, 8, 12, 15, 18, 22, 25, 30],
  value: [400, 450, 535, 640, 815, 970, 1160, 1465, 1720, 2300],
};
const test = { years: [3, 10, 20, 28] };

console.table(data.years.map((y, i) => ({ years: y, value: data.value[i] })));

const X = data.years.map((x) => [x]);
const Y = data.value.map((v) => [Math.log(v)]);

const reg = new MLR(X, Y);

const r = reg.weights[0][0];
const a = Math.exp(reg.weights[1][0]);
console.log(`fit: value = ${a.toFixed(4)} * exp(${r.toFixed(6)} * years)`);
console.log(`     implied annual appreciation: ${((Math.exp(r) - 1) * 100).toFixed(2)}%`);

const predictions = test.years.map((x) => a * Math.exp(r * x));
console.table(test.years.map((x, i) => ({ years: x, value: predictions[i] })));

const yearGrid = linspace(Math.min(...data.years), Math.max(...data.years), 200);
const curve = yearGrid.map((x) => a * Math.exp(r * x));

writePlotlyHtml(
  new URL('./02d_exponential_plot.html', import.meta.url).pathname,
  'Exponential Regression - years held vs value',
  [
    { type: 'scatter', mode: 'markers', name: 'training',
      x: data.years, y: data.value, marker: { color: 'magenta', symbol: 'star', size: 12 } },
    { type: 'scatter', mode: 'lines', name: 'exponential fit',
      x: yearGrid, y: curve, line: { color: 'blue' } },
    { type: 'scatter', mode: 'markers', name: 'predicted',
      x: test.years, y: predictions, marker: { color: 'red', size: 10 } },
  ],
  { xaxis: { title: 'years held' }, yaxis: { title: 'property value ($k)' } },
);

console.log('\nWrote 02d_exponential_plot.html');
