import * as React from "react";
import "./Samples.css";
import { Slider, InputNumber, Row, Col, Switch } from 'antd';
// import "../cache/heatmaps_nobbox2017/pos.json";
let dots = require("../cache/heatmaps_nobbox2017/pos.json");
let truePreds = require("../cache/true_preds2017.json");
export interface Dot {
    pos: [number, number];
    pred: number;
    [key: string]: any
}
export interface States {
    th: number;
    allSamples: boolean;
    // selectedIDs:number[];
    selectBox:{x:number, y:number, w:number, h:number}
}

export interface Props {
    onSelectIDs: (ids: number[]) => void
}

const THRED = 0.15
let width = window.innerWidth * 0.75
let height = (window.innerHeight - 70) * 0.4
let margin = 20
let newdots: Dot[] = dots.map((p: Dot, i: number) => {
    return {
        pos: [p.pos[0] * (width - 3 * margin), p.pos[1] * (height - 2 * margin)],
        pred: Math.abs(p.pred - truePreds[i])
    }
})

export default class Samples extends React.Component<Props, States> {
    public selectedIDs:number[]
    constructor(props: any) {
        super(props)
        this.state = {
            th: THRED,
            allSamples: true,
            // selectedIDs:[],
            selectBox:{x:0, y:0, w:0,h:0}
        }
        this.selectedIDs=[]
        this.changeTH = this.changeTH.bind(this)
        this.onSelectIDs = this.onSelectIDs.bind(this)
        this.startSelect = this.startSelect.bind(this)
        this.onSelect = this.onSelect.bind(this)
        this.endSelect = this.endSelect.bind(this)
    }
    changeTH(value: number) {
        console.info(value)
        this.setState({
            th: value
        });
    }
    onSelectIDs(ids: number[]): void {
        this.selectedIDs.concat(ids)
        this.props.onSelectIDs(this.selectedIDs)
        this.selectedIDs=[]
        // this.setState({selectedIDs: ids})
    }
    startSelect(e:React.MouseEvent<any>){
        document.addEventListener("mousemove", this.onSelect)
        this.setState({selectBox:{x:e.clientX, y:e.clientY-70, w:0, h:0}})
    }
    onSelect(e:any){
        let {x,y, w, h} = this.state.selectBox
        newdots.forEach((dot:Dot,i:number)=>{
            if(
                dot.pos[0]>x && 
                dot.pos[0]<x+w && 
                dot.pos[1]>y && 
                dot.pos[1]<y+h &&
                this.selectedIDs.indexOf(i)==-1
            ){
                this.selectedIDs.push(i)
            }
        })
        this.setState({selectBox:{
            x:e.clientX-x>0?x:x-w, 
            y:e.clientY-y>0?y:y-h, 
            w:Math.abs(e.clientX-x), 
            h:Math.abs((e.clientY-70)-y)
        }})

    }
    endSelect(e:React.MouseEvent<any>){
        document.removeEventListener("mousemove", this.onSelect)
        // this.setState({selectBox:{x:0, y:0, w:0, h:0}})
        this.props.onSelectIDs(this.selectedIDs)
        this.selectedIDs=[]
    }
    render() {
        let { allSamples } = this.state
        let {x,y,w,h} =  this.state.selectBox
        return (
            <div>
                <Switch
                    className="sampleSwitch"
                    checkedChildren=""
                    unCheckedChildren=""
                    defaultChecked
                    onChange={() => { this.setState({ allSamples: !allSamples }) }}
                />

                <svg className="samples" width={width - margin} height={height} 
                onMouseDown={this.startSelect}
                onMouseUp={this.endSelect}
                >
                    <rect className='selectBox' width={w} height={h} x={x} y={y}></rect>

                    <g className="samples" transform={`translate (${margin}, ${margin})`}>
                        {newdots.map((dot: Dot, i: number) => {

                            let aboveTH: boolean = (dot.pred >= this.state.th)
                            let selected:boolean = this.selectedIDs.indexOf(i) !=-1

                            if (allSamples || (!allSamples && aboveTH)) {
                                return <circle key={i} cx={dot.pos[0]} cy={dot.pos[1]}
                                    r={aboveTH ?(selected? 5:4) : 3}
                                    fill={aboveTH ?(selected? "#027dd6":"#49a9ee"): "gray"}
                                    // onClick={() => this.onSelectIDs([i])}
                                    >
                                </circle>
                            } else {
                                return <circle />
                            }

                        })}
                    </g>
                </svg>

                <Row className="threshold">
                    <Col span={2}><div style={{ textAlign: "center" }}>Threshold</div></Col>
                    <Col span={8}>
                        <Slider min={0} max={1} value={this.state.th} onChange={this.changeTH} step={0.01} />
                    </Col>
                    <Col span={4}>
                        <InputNumber
                            min={0}
                            max={1}
                            step={0.01}
                            style={{ marginLeft: 16 }}
                            value={this.state.th}
                            onChange={this.changeTH}
                        />
                    </Col>
                </Row>
            </div>)
    }
}