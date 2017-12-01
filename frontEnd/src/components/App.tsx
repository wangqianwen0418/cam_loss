
import * as React from 'react';
import "./App.css";
import SiderBar from "../containers/SideBar";
import Graph from "./Graph";

import { Row, Col} from 'antd';



class App extends React.Component{
    render() {

        return (
            <div className="app">
                <Row>
                  <Col span={24}>
                  <div className="header">Visual DNN</div>
                  </Col>
                </Row>
                <Row>
                    <Col span={3}><SiderBar /></Col>
                    <Col span={21}>
                        <Graph/>
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