import * as React from "react";
import "./Annotation.css";
import { Switch } from 'antd';



export interface Props {
    ids: number[]
    bbox:boolean
    imgset:"train"|"val"
}
export interface State {
    // bbox:boolean;
    heatmap:boolean;
}
const margin = 10, col = 4, row = 5;
export default class Annotation extends React.Component<Props, State>{
    constructor(props:Props){
        super(props)
        this.state = {
            // bbox:false, 
            heatmap: true
        }
    }
    render() {
        let {heatmap} = this.state
        let {bbox, imgset} = this.props
        let dots = require(`../cache/heatmaps_${bbox?"bbox":"nobbox"}2017_${imgset}/pos.json`);
        let truePreds = require(`../cache/true_preds2017_${imgset}.json`);
        return (
            <div className="annotation">
                <Switch
                    className="heatmapSwitch"
                    checkedChildren="heatmap"
                    unCheckedChildren="origin"
                    defaultChecked
                    onChange={() => { this.setState({ heatmap: !heatmap }) }}
                />
                {/* <Switch
                    className="bboxSwitch"
                    checkedChildren="bbox"
                    unCheckedChildren="no bbox"
                    onChange={() => { this.setState({ bbox: !bbox }) }}
                /> */}
                {this.props.ids.map((id: number) => {
                    let src:string = heatmap?(bbox?`heatmaps_bbox2017_${imgset}`:`heatmaps_nobbox2017_${imgset}`):`2017_${imgset}`
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