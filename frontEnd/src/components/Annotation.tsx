import * as React from "react";
import "./Annotation.css";
import { Switch, Modal } from 'antd';



export interface Props {
    ids: number[]
    bbox: boolean
    imgset: "train" | "val",
    onChangeBBox: (bbox:boolean)=>void
}
export interface State {
    bbox:boolean;
    heatmap: boolean;
    visible: boolean;
}
const margin = 5, col = 4, row = 5;
export default class Annotation extends React.Component<Props, State>{
    public selectedID: number = 0;
    constructor(props: Props) {
        super(props)
        this.state = {
            bbox:this.props.bbox, 
            heatmap: true,
            visible: false
        }
    }
    render() {
        let { bbox, heatmap } = this.state
        let { imgset } = this.props
        let dots = require(`../cache/heatmaps_${bbox ? "bbox" : "nobbox"}2017_${imgset}/pos.json`);
        let truePreds = require(`../cache/true_preds2017_${imgset}.json`);

        let src: string = heatmap ? (bbox ? `heatmaps_bbox2017_${imgset}` : `heatmaps_nobbox2017_${imgset}`) : `2017_${imgset}`
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

                    return <img key={id}
                        style={{ paddingLeft: margin }}
                        width={(window.innerWidth - 10) * 0.75 / row}
                        height={(window.innerHeight - 70) * 0.6 / col - margin * 2}
                        src={`./cache/${src}/${id}.jpg`}
                        title={`id:${id}, pred:${dots[id].pred.toFixed(4)}, truth:${truePreds[id]}`}
                        onClick={() => { this.setState({ visible: true }); this.selectedID = id }}
                    >
                    </img>
                })}

                <Modal
                    title={`ID:${this.selectedID}`}
                    visible={this.state.visible}
                    onCancel={() => { this.setState({ visible: false }) }}
                    width={window.innerWidth * 0.6}
                    footer={[
                        <Switch
                            checkedChildren="heatmap"
                            unCheckedChildren="origin"
                            defaultChecked
                            onChange={() => { this.setState({ heatmap: !heatmap }) }}
                        />, 
                        <Switch
                            checkedChildren="bbox"
                            unCheckedChildren="nobbox"
                            onChange={() => {  this.setState({bbox:!bbox}) ;this.props.onChangeBBox(!bbox) }}
                        />,
                        <span>
                            {`id:${this.selectedID}, 
                        pred:${dots[this.selectedID].pred.toFixed(4)}, 
                        truth:${truePreds[this.selectedID]}`}
                        </span>
                    ]}
                > <img width={window.innerWidth * 0.6 - 2 * 16} src={`./cache/${src}/${this.selectedID}.jpg`}></img>
                </Modal>
            </div>)
    }
}