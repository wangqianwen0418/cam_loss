import * as React from "react";
import "./Annotation.css";
import { Switch } from 'antd';
export interface Props {
    ids: number[]
}
export interface State {
    bbox:boolean;
    heatmap:boolean;
}
const margin = 5, col = 2, row = 3;
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
                        style={{ marginRight: margin }}
                        width={window.innerWidth * 0.75 / row - margin * 2.5}
                        height={(window.innerHeight - 70) * 0.6 / col - margin * 2}
                        src={`./cache/${src}/${id}.jpg`}
                    >
                    </img>
                })}
            </div>)
    }
}