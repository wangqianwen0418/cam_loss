import * as React from 'react';
import * as ReactDOM from 'react-dom';
import App from './components/App';
import { createStore } from 'redux';
import { reducer } from './reducers/index';
import { StoreState } from './types/index';
import { Provider } from 'react-redux';
import registerServiceWorker from './registerServiceWorker';
import './index.css';

import 'antd/dist/antd.css';

let initIDs:number[]=[]
for(let i =0;i<100;i++){
  initIDs.push(Math.round(Math.random()*2088))
}
let initState:StoreState = {
  model:{node:[]}, 
  ids:initIDs}
const store = createStore<StoreState>(reducer, initState );

console.info(Array(100).map((e:any,i:number)=>Math.round(Math.random()*2000)))
ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById('root') as HTMLElement
);
registerServiceWorker();
