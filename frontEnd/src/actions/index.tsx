import * as constants from '../constants'
import {Model} from "../types"

// export interface IncrementEnthusiasmAction {
//     type: constants.INCREMENT_ENTHUSIASM;
// }

// export interface DecrementEnthusiasmAction {
//     type: constants.DECREMENT_ENTHUSIASM;
// }


// export function incrementEnthusiasm(): IncrementEnthusiasmAction {
//     return {
//         type: constants.INCREMENT_ENTHUSIASM
//     }
// }

// export function decrementEnthusiasm(): DecrementEnthusiasmAction {
//     return {
//         type: constants.DECREMENT_ENTHUSIASM
//     }
// }


// 
export interface ImportModelAction {
    type: constants.IMPORT_MODEL,
    model:Model

}
export function importModel(json:Model):ImportModelAction{
    return {
        type:constants.IMPORT_MODEL,
        model:json
    }
}

export interface SelectIDsAction {
    type: constants.SELECT_IDS,
    ids: number[]

}
export function selectIDs(ids:number[]):SelectIDsAction{
    return {
        type:constants.SELECT_IDS,
        ids
    }
}

// export type EnthusiasmAction = IncrementEnthusiasmAction | DecrementEnthusiasmAction
export type AllActions = ImportModelAction|SelectIDsAction
