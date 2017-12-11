import ControlPanel from '../components/ControlPanel';
import * as actions from '../actions/';
import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps(state: StoreState) {
    return {
        // ids:state.ids
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.AllActions>) {
    return {
        onChangeImgSet: (imgset:"train"|"val") => {dispatch(actions.changeImgSet(imgset))},
        onChangeBBox: (f:boolean) => {dispatch(actions.changeBBox(f))}
        // onImportModel: (json:any) => {dispatch(actions.importModel(json))}
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ControlPanel);