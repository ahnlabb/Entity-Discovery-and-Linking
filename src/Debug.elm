module Main exposing (..)

import Browser
import Http
import Url.Builder as Url
import Dict
import Element exposing (Element, el, text, row, column, alignRight, fill, width, height, rgb255, spacing, centerY, padding, none, px)
import Element.Input as Input
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Html exposing (select, option)
import Html.Events exposing (onInput)
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


type Model
    = Loading
    | Error Http.Error
    | Done Page


type alias Page =
    { docs : List Document
    , selection : Maybe Int
    }


type alias Document =
    { text : String
    , entities : List Entity
    }


type alias Entity =
    { start : Int
    , stop : Int
    , class : String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( Loading, Cmd.none )



-- UPDATE


type Msg
    = NewDocuments (Result Http.Error (List Document))
    | NewSelection String


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model ) of
        ( NewDocuments result, Loading ) ->
            case result of
                Ok docs ->
                    ( Done { docs = docs, selection = Nothing }, Cmd.none )

                Err e ->
                    ( Error e, Cmd.none )

        ( NewSelection string, Done page ) ->
            ( Done { page | selection = String.toInt string }, Cmd.none )

        ( _, _ ) ->
            ( model, Cmd.none )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions _ =
    Sub.none



-- VIEW


view model =
    case model of
        Done page ->
            Element.layout []
                (body page)

        Loading ->
            Element.layout []
                (none)

        Error e ->
            Element.layout []
                (none)


body : Page -> Element Msg
body { docs, selection } =
    column [ width fill, spacing 30 ]
        [ selectDoc (List.length docs)
        , resultView docs selection
        ]


selectDoc : Int -> Element Msg
selectDoc numberOfDocs =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Element.html (select [ onInput NewSelection ] (List.range 1 numberOfDocs |> List.map intToOption)))


intToOption int =
    option [] [ Html.text (String.fromInt int) ]


resultView docs selection =
    let
        get list index =
            List.drop (index - 1) list |> List.head
    in
        case (selection |> Maybe.andThen (get docs)) of
            Just doc ->
                let
                    fontSz =
                        20

                    pos x y tag attrs =
                        tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

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

                    mark string start end =
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

                    markEntity { start, stop, class } =
                        mark class start stop
                in
                    el [ width fill ]
                        (Element.html
                            (Svg.svg
                                [ SAttr.width "600"
                                , SAttr.height "600"
                                , SAttr.viewBox "0 0 600 600"
                                ]
                                ([ svgText fontSz (toFloat margin) (toFloat lineSeparation) doc.text ]
                                    ++ (List.map markEntity doc.entities)
                                )
                            )
                        )

            Nothing ->
                none


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
    Url.absolute [ "gold" ] []


getDocuments : String -> Cmd Msg
getDocuments text =
    Http.get
        { url = localApi
        , expect = Http.expectJson NewDocuments (list documentDecoder)
        }


documentDecoder =
    Decode.succeed Document
        |> required "text" string
        |> required "gold" (list entityDecoder)


entityDecoder =
    Decode.succeed Entity
        |> required "start" int
        |> required "stop" int
        |> required "class" string
