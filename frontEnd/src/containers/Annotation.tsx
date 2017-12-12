import Annotation from '../components/Annotation';
import * as actions from '../actions/';
import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps(state: StoreState) {
    return {
        ids:state.ids,
        bbox: state.bbox,
        imgset: state.imgset
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.AllActions>) {
    return {
        // onImportModel: (json:any) => {dispatch(actions.importModel(json))}
        onChangeBBox: (bbox:boolean) => {dispatch(actions.changeBBox(bbox))}
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Annotation);