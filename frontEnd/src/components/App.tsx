
import * as React from 'react';
import "./App.css";
import Samples from "../containers/Samples";
import Annotation from "../containers/Annotation";
import Model from "../containers/Model";
import ControlPanel from "../containers/ControlPanel";

import { Row, Col} from 'antd';



class App extends React.Component{
    render() {

        return (
            <div className="app">
                <Row>
                  <Col span={24}>
                  <div className="header">Look  Here</div>
                  </Col>
                </Row>
                <Row>
                    <Col span={18}>
                        <Row><ControlPanel/></Row>
                        <Row><Samples /></Row>
                        <Row><Annotation/></Row>
                    </Col>
                    <Col span={6}>
                        <Model/>
                    </Col>
                </Row>
            </div>
        );
    }
}

export default App;

// helpers

// function getExclamationMarks(numChars: number) {
//     return Array(numChars + 1).join('!');
// }