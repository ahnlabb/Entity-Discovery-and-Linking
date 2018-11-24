module Main exposing (..)

import Browser
import Http
import Url.Builder as Url
import Dict
import Element exposing (Element, el, text, row, alignRight, fill, width, height, rgb255, spacing, centerY, padding, none, px)
import Element.Input as Input
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Json.Decode as Decode exposing (Decoder, int, string, dict, list)
import Json.Decode.Pipeline exposing (required, custom)
import Json.Encode as Encode
import Time
import Svg
import Svg.Attributes as SAttr


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type alias Model =
    { text : String
    , result : String
    , change : Change
    , docforia : Maybe Docforia
    , error : Maybe Http.Error
    }


type alias Docforia =
    { text : String
    , properties : Properties
    , edges : List Edges
    , nodes : List Nodes
    }


type alias Edges =
    { layer : String
    , properties : List Properties
    , connections : List Int
    }


type alias Nodes =
    { layer : String
    , properties : List Properties
    , ranges : List Int
    }


type alias Properties =
    Dict.Dict String String


type Change
    = Waiting
    | Changed
    | Unchanged


init : () -> ( Model, Cmd Msg )
init _ =
    ( Model "" "" Waiting Nothing Nothing, Cmd.none )



-- UPDATE


type Msg
    = NewDocforia (Result Http.Error Docforia)
    | EditedText String
    | NewEl (Result Http.Error String)
    | Tick Time.Posix


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NewDocforia result ->
            case result of
                Ok data ->
                    ( { model | docforia = Just data }, Cmd.none )

                Err e ->
                    ( { model | error = Just e }, Cmd.none )

        EditedText newText ->
            ( { model | text = newText, change = Changed }, Cmd.none )

        NewEl result ->
            case result of
                Ok newResult ->
                    ( { model | result = newResult }, Cmd.none )

                Err _ ->
                    ( model, Cmd.none )

        Tick _ ->
            case model.change of
                Waiting ->
                    ( model, Cmd.none )

                Changed ->
                    ( { model | change = Unchanged }, Cmd.none )

                Unchanged ->
                    ( { model | change = Waiting }, getCoreNLP "en" model.text )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions _ =
    Time.every 500 Tick



-- VIEW


view model =
    Element.layout []
        (body model)


body : Model -> Element Msg
body model =
    row [ width fill, spacing 30 ]
        [ textInput model.text
        , resultView model
        ]


textInput : String -> Element Msg
textInput text =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Input.multiline
            [ height (px 600) ]
            { label = Input.labelHidden ""
            , onChange = EditedText
            , placeholder = Nothing
            , spellcheck = False
            , text = text
            }
        )


layer =
    (++) "se.lth.cs.docforia.graph.text."


getLayer layerName list =
    case list of
        h :: rest ->
            if h.layer == layer layerName then
                Just h
            else
                getLayer layerName rest

        [] ->
            Nothing


inTwos list =
    case list of
        fst :: snd :: rest ->
            ( fst, snd ) :: (inTwos rest)

        _ ->
            []


resultView : Model -> Element msg
resultView model =
    case model.docforia of
        Just doc ->
            let
                fontSz =
                    20

                pos x y tag attrs =
                    tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

                getNodeRanges layerName =
                    getLayer layerName doc.nodes |> Maybe.map (.ranges >> inTwos) |> Maybe.withDefault []

                sentences =
                    getNodeRanges "Sentence"

                tokens =
                    getNodeRanges "Token"

                tokenProps =
                    getLayer "Token" doc.nodes |> Maybe.map .properties |> Maybe.withDefault []

                textStyle sz =
                    SAttr.style ("font-size: " ++ (String.fromInt sz) ++ "px;font-family: 'Source Code Pro', monospace;")

                lineStyle =
                    SAttr.style "stroke:rgb(100,100,100);stroke-width:2"

                svgText sz x y text =
                    pos x y Svg.text_ [ textStyle sz ] [ Svg.text text ]

                lineSeparation =
                    50

                toSvg i ( start, end ) =
                    svgText fontSz (toFloat margin) (toFloat (i + 1) * lineSeparation) (String.slice start end doc.text)

                line x1 y1 x2 y2 =
                    List.map2 (\f x -> String.fromInt x |> f) [ SAttr.x1, SAttr.y1, SAttr.x2, SAttr.y2 ] [ x1, y1, x2, y2 ]
                        |> (::) lineStyle
                        |> \x -> Svg.line x []

                mark string ( start, end ) =
                    let
                        midX =
                            (toFloat (start + end)) / 2 * charWidth

                        annoSz =
                            12

                        scaleFactor =
                            toFloat annoSz / fontSz

                        scale =
                            toFloat >> (*) scaleFactor

                        length =
                            String.length string

                        x =
                            margin + midX - (scale length) / 2 * charWidth

                        xStr =
                            String.fromFloat x

                        width =
                            length * charWidth + 8 |> scale |> String.fromFloat

                        yStr =
                            lineSeparation + 4 |> String.fromInt

                        height =
                            charHeight + 8 |> scale |> String.fromFloat

                        y =
                            lineSeparation + 4 + scale charHeight

                        colorFromPos posStr =
                            case posStr of
                                "DT" ->
                                    "yellow"

                                "JJ" ->
                                    "purple"

                                "TO" ->
                                    "purple"

                                "VBD" ->
                                    "blue"

                                "VBN" ->
                                    "blue"

                                "NNP" ->
                                    "green"

                                "CD" ->
                                    "green"

                                _ ->
                                    "red"
                    in
                        Svg.g []
                            [ Svg.rect
                                [ SAttr.x xStr
                                , SAttr.width width
                                , SAttr.y yStr
                                , SAttr.height height
                                , SAttr.rx "5"
                                , SAttr.ry "5"
                                , SAttr.style ("fill:" ++ colorFromPos string ++ ";stroke:black;stroke-width:1;opacity:0.5")
                                ]
                                []
                            , svgText annoSz (x + scale 4) (y + scale 4) string
                            ]

                charWidth =
                    12

                charHeight =
                    15

                margin =
                    50

                markToken props slice =
                    mark (Dict.get "pos" props |> Maybe.withDefault "") slice
            in
                el [ width fill ]
                    (Element.html
                        (Svg.svg
                            [ SAttr.width "600"
                            , SAttr.height "600"
                            , SAttr.viewBox "0 0 600 600"
                            ]
                            ((List.indexedMap toSvg sentences)
                                ++ List.map2 markToken tokenProps tokens
                            )
                        )
                    )

        Nothing ->
            el [ width fill ] (errorString model.error |> text)


errorString error =
    case error of
        Nothing ->
            "----"

        Just (Http.BadBody str) ->
            "BadBody: " ++ str

        Just (Http.BadStatus code) ->
            "BadStatus: " ++ (String.fromInt code)

        Just (Http.BadUrl str) ->
            "BadUrl: " ++ str

        Just Http.NetworkError ->
            "NetworkError"

        Just Http.Timeout ->
            "Timeout"



-- HTTP


localApi : String
localApi =
    Url.absolute [ "el" ] []


getEl : String -> Cmd Msg
getEl text =
    Http.post
        { url = localApi
        , body = Http.jsonBody (Encode.string text)
        , expect = Http.expectJson NewEl elDecoder
        }


elDecoder =
    string


getCoreNLP : String -> String -> Cmd Msg
getCoreNLP lang string =
    let
        postData =
            Http.jsonBody (Encode.string string)
    in
        Http.post
            { url = localApi
            , body = postData
            , expect = Http.expectJson NewDocforia docforiaDecoder
            }


docforiaDecoder =
    Decode.field "DM10" docforiaHelper


docforiaHelper =
    Decode.succeed Docforia
        |> required "text" string
        |> required "properties" propertiesDecoder
        |> required "edges" (list edgesDecoder)
        |> required "nodes" (list nodesDecoder)


propertiesDecoder =
    dict string


decodeFirst =
    Decode.index 0


edgesHelper =
    Decode.field "edges" << decodeFirst


nodesHelper =
    Decode.field "nodes" << decodeFirst


edgesDecoder =
    Decode.succeed Edges
        |> required "layer" string
        |> custom (edgesHelper <| Decode.field "properties" <| list propertiesDecoder)
        |> custom (edgesHelper <| Decode.field "connections" <| list int)


nodesDecoder =
    Decode.succeed Nodes
        |> required "layer" string
        |> custom (nodesHelper <| Decode.field "properties" <| list propertiesDecoder)
        |> custom (nodesHelper <| Decode.field "ranges" <| list int)
