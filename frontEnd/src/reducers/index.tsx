// src/reducers/index.tsx

import { AllActions } from '../actions';
import { StoreState } from '../types/index';
import { IMPORT_MODEL, SELECT_IDS } from '../constants/index';

export function reducer(state: StoreState, action: AllActions): StoreState {
  switch (action.type) {
    // case INCREMENT_ENTHUSIASM:
    
    //   return { ...state, enthusiasmLevel:state.enthusiasmLevel+1 };
    // case DECREMENT_ENTHUSIASM:
    //   return { ...state, enthusiasmLevel:state.enthusiasmLevel-1 };
    case IMPORT_MODEL:    
      return { ...state, model:action.model}
    case SELECT_IDS:
      return {...state, ids:action.ids}
    default:
      return state;
  }
}