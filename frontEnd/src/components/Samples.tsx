import * as React from "react";
import "./Samples.css";
import { Slider, Switch } from 'antd';
// import "../cache/heatmaps_nobbox2017/pos.json";

export interface Dot {
    pos: [number, number];
    pred: number;
    conf:number;
    [key: string]: any
}
export interface States {
    th: number;
    allSamples: boolean;
    // selectedIDs:number[];
    selectBox: { x: number, y: number, w: number, h: number }
}

export interface Props {
    bbox:boolean,
    imgset:"train"|"val",
    onSelectIDs: (ids: number[]) => void
}

const THRED = 0.15
let width = window.innerWidth * 0.75 - 100
let height = (window.innerHeight - 70) * 0.4
let margin = 20
const topPadding = 120



export default class Samples extends React.Component<Props, States> {
    public selectedIDs: number[];newdots:Dot[]
    constructor(props: any) {
        super(props)
        this.state = {
            th: THRED,
            allSamples: true,
            // selectedIDs:[],
            selectBox: { x: 0, y: 0, w: 0, h: 0 }
        }
        this.selectedIDs = []
        this.changeTH = this.changeTH.bind(this)
        // this.onSelectIDs = this.onSelectIDs.bind(this)
        this.startSelect = this.startSelect.bind(this)
        this.onSelect = this.onSelect.bind(this)
        this.endSelect = this.endSelect.bind(this)

        let dots = require("../cache/heatmaps_nobbox2017_train/pos.json");
        let truePreds = require("../cache/true_preds2017_train.json");
        // dots.splice(1987, 1)
        this.newdots = dots.map((p: Dot, i: number) => {
            return {
                pos: [p.pos[0] * (width - 3 * margin), p.pos[1] * (height - 2 * margin)],
                id:i,
                pred:p.pred,
                conf: Math.abs(p.pred - truePreds[i])
            }
        })
       
       
    }
    changeTH(value: number) {
        console.info(value)
        this.setState({
            th: value
        });
    }
    // onSelectIDs(ids: number[]): void {
    //     this.selectedIDs.concat(ids)
    //     this.props.onSelectIDs(this.selectedIDs)
    //     this.selectedIDs=[]
    //     // this.setState({selectedIDs: ids})
    // }
    startSelect(e: React.MouseEvent<any>) {
        document.addEventListener("mousemove", this.onSelect)
        this.setState({ selectBox: { x: e.pageX - margin, y: e.pageY - topPadding - margin, w: 0, h: 0 } })
    }
    onSelect(e: any) {
        let { x, y } = this.state.selectBox
        let sx: number = e.pageX - margin, sy: number = e.pageY - topPadding - margin

        this.setState({
            selectBox: {
                x: sx - x >= 0 ? x : sx,
                y: sy - y >= 0 ? y : sy,
                w: Math.abs(sx - x),
                h: Math.abs(sy - y)
            }
        })

    }
    endSelect(e: React.MouseEvent<any>) {
        document.removeEventListener("mousemove", this.onSelect)
        let { x, y, w, h } = this.state.selectBox 

        this.newdots.filter((dot: Dot) => { return this.state.allSamples || dot.conf >= this.state.th })
        .forEach((dot: Dot, i: number) => {
            let px: number = dot.pos[0], py: number = dot.pos[1]
            if (
                px > x &&
                px < x + w &&
                py > y &&
                py < y + h &&
                this.selectedIDs.indexOf(dot.id) == -1
            ) {
                this.selectedIDs.push(dot.id)
            }
        })
        // this.setState({selectBox:{x:0, y:0, w:0, h:0}})
        if (this.selectedIDs.length > 0 || (w > 2 && h > 2)) {
            this.props.onSelectIDs(this.selectedIDs)
            this.selectedIDs = []
        }
    }
    componentWillReceiveProps(nextProps:Props){
        let {bbox, imgset} = nextProps
        if(imgset!=this.props.imgset||bbox!=this.props.bbox){
            
            let dots = require(`../cache/heatmaps_${bbox?"bbox":"nobbox"}2017_${imgset}/pos.json`);
            let truePreds = require(`../cache/true_preds2017_${imgset}.json`);
            
            this.newdots = dots.map((p: Dot, i: number) => {
                return {
                    pos: [p.pos[0] * (width - 3 * margin), p.pos[1] * (height - 2 * margin)],
                    id:i,
                    conf: Math.abs(p.pred - truePreds[i])
                }
            })
        }
        
    }
    render() {
        let { allSamples } = this.state
        let { x, y, w, h } = this.state.selectBox
        return (
            <div>
                <Switch
                    className="sampleSwitch"
                    checkedChildren="all samples"
                    unCheckedChildren="above th"
                    defaultChecked
                    onChange={() => { this.setState({ allSamples: !allSamples }) }}
                />

                <svg className="samples" width={width - margin} height={height}
                    onMouseDown={this.startSelect}
                    onMouseUp={this.endSelect}
                    // onMouseOut={this.endSelect}
                >
                    <g transform={`translate (${margin}, ${margin})`}>
                        <rect className='selectBox' width={w} height={h} x={x} y={y}></rect>

                        <g className="samples" >

                            {this.newdots.filter((dot: Dot) => { return (allSamples || (dot.conf >= this.state.th)) })
                                .map((dot: Dot) => {

                                    let aboveTH: boolean = (dot.conf >= this.state.th)
                                    let selected: boolean = this.selectedIDs.indexOf(dot.id) != -1
                                    return <circle
                                        className="dot"
                                        key={dot.id} cx={dot.pos[0]} cy={dot.pos[1]}
                                        r={aboveTH ? (selected ? 3 : 3) : 3}
                                        fill={aboveTH ? "#49a9ee" : "#ccc"}
                                    // onClick={() => this.onSelectIDs([i])}
                                    >
                                        <title>id:{dot.id}</title>
                                    </circle>
                                })}
                        </g>
                    </g>
                </svg>
                <div className="threshold">
                    {/* <InputNumber
                        min={0}
                        max={1}
                        step={0.01}
                        style={{ marginLeft: 16 }}
                        value={this.state.th}
                        onChange={this.changeTH}
                    /> */}
                    <Slider
                        vertical
                        min={0} max={1}
                        value={this.state.th}
                        onChange={this.changeTH}
                        step={0.01} />
                </div>

                {/* <Row className="threshold">
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
                </Row> */}
            </div>)
    }
}