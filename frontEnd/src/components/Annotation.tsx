import * as React from "react";
import "./Annotation.css";
import { Switch } from 'antd';
let dots = require("../cache/heatmaps_nobbox2017/pos.json");
let truePreds = require("../cache/true_preds2017.json");

export interface Props {
    ids: number[]
}
export interface State {
    bbox:boolean;
    heatmap:boolean;
}
const margin = 10, col = 4, row = 5;
export default class Annotation extends React.Component<Props, State>{
    constructor(props:Props){
        super(props)
        this.state = {
            bbox:false, 
            heatmap: true
        }
    }
    render() {
        let {bbox, heatmap} = this.state
        return (
            <div className="annotation">
                <Switch
                    className="heatmapSwitch"
                    checkedChildren="heatmap"
                    unCheckedChildren="origin"
                    defaultChecked
                    onChange={() => { this.setState({ heatmap: !heatmap }) }}
                />
                <Switch
                    className="bboxSwitch"
                    checkedChildren="bbox"
                    unCheckedChildren="no bbox"
                    onChange={() => { this.setState({ bbox: !bbox }) }}
                />
                {this.props.ids.map((id: number) => {
                    let src:string = heatmap?(bbox?"heatmaps_bbox2017":"heatmaps_nobbox2017"):"2017"
                    return <img key={id}
                        style={{ paddingRight: margin }}
                        width={(window.innerWidth - 10) * 0.75 / row -3}
                        height={(window.innerHeight - 70) * 0.6 / col - margin * 2}
                        src={`./cache/${src}/${id}.jpg`}
                        title={`id:${id}, pred:${dots[id].pred.toFixed(4)}, truth:${truePreds[id]}`}
                    >
                    </img>
                })}
            </div>)
    }
}