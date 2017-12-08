import Samples from '../components/Samples';
import * as actions from '../actions/';
// import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps() {
    return {
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.AllActions>) {
    return {
        onSelectIDs: (ids:number[]) => {dispatch(actions.selectIDs(ids))}
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Samples);