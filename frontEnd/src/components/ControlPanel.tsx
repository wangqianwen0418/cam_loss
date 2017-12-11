import * as React from "react";
import { Switch, Icon } from 'antd';
import "./ControlPanel.css";
export interface Props{
    onChangeImgSet: (imgset:"train"|"val")=>void
    onChangeBBox: (f:boolean)=> void
}

export default class ControlPanel extends React.Component<Props, {}>{
    constructor(props:Props){
        super(props)

    }
    render(){
        return <div className="controlPanel">
            <Icon className="setting" type="setting" style={{fontSize:"1.5em"}}/>
            <Switch
                    className="imgSetSwitch setting"
                    checkedChildren="train"
                    unCheckedChildren="val"
                    defaultChecked
                    onChange={(v:boolean) => { this.props.onChangeImgSet(v?"train":"val") }}
                />
            <Switch
                    className="boxSwitch setting"
                    checkedChildren="with bbox"
                    unCheckedChildren="no bbox"
                    onChange={(v:boolean) => { this.props.onChangeBBox(v) }}
                />
            </div>
    }
}