import * as React from "react";
import "./Samples.css";
import { Slider, InputNumber, Row, Col } from 'antd';
// import "../cache/heatmaps_nobbox2017/pos.json";
let dots = require("../cache/heatmaps_nobbox2017/pos.json");
let truePreds = require("../cache/true_preds2017.json");
export interface Dot {
    pos: [number, number];
    pred: number;
    [key: string]: any
}
export interface States{
    th:number
}

const THRED = 0.15
export default class Samples extends React.Component<{},States> {
    constructor(props:any){
        super(props)
        this.state={
            th:THRED
        }
        this.onChange = this.onChange.bind(this)
    }
    onChange(value:number){
        console.info(value)
        this.setState({
            th: value
          });
    }
    click(e:React.MouseEvent<any>):void{
        console.info(e)
    }

    render() {
        let margin = 20
        let width = window.innerWidth * 0.75
        let height = (window.innerHeight - 70) * 0.4
        let newdots:Dot[] = dots.map((p: Dot, i: number) => {
            return {
                pos: [p.pos[0] * (width - 3 * margin), p.pos[1] * (height - 2 * margin)],
                pred: Math.abs(p.pred - truePreds[i])
            }
        })
        return (
            <div>
                <svg className="samples" width={width - margin} height={height}>
                    <g className="samples" transform={`translate (${margin}, ${margin})`}>
                        {newdots.map((dot: Dot, i: number) => {
                            return <circle key={i} cx={dot.pos[0]} cy={dot.pos[1]} 
                            r={dot.pred >= this.state.th ?5:3} 
                            fill={dot.pred >= this.state.th ? "#49a9ee" : "gray"}
                            onClick={this.click}>
                            </circle>
                        })}
                    </g>
                </svg>
                <Row className="threshold">
                    <Col span={2}><div style={{textAlign:"center"}}>Threshold</div></Col>
                    <Col span={8}>
                        <Slider min={0} max={1} value={this.state.th} onChange={this.onChange} step={0.01} />
                    </Col>
                    <Col span={4}>
                        <InputNumber
                            min={0}
                            max={1}
                            step={0.01}
                            style={{ marginLeft: 16 }}
                            value={this.state.th}
                            onChange={this.onChange}
                        />
                    </Col>

                </Row>
            </div>)
    }
}