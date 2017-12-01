import SideBar from '../components/SideBar';
import * as actions from '../actions/';
// import { StoreState } from '../types/index';
import { connect, Dispatch } from 'react-redux';

export function mapStateToProps() {
    return {
    };
}

export function mapDispatchToProps(dispatch: Dispatch<actions.ImportModelAction>) {
    return {
        onImportModel: (json:any) => {dispatch(actions.importModel(json))}
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(SideBar);