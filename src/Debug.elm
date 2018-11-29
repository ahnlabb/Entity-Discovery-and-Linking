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
import Html.Attributes as HAttr exposing (style)
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
    { docs : Dict.Dict String Document
    , selection : Maybe String
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
    ( Loading, getDocuments )



-- UPDATE


type Msg
    = NewDocuments (Result Http.Error (Dict.Dict String Document))
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
            ( Done { page | selection = Just string }, Cmd.none )

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
                (el [] (text "loading"))

        Error e ->
            Element.layout []
                (el [] (text (errorString e)))


body : Page -> Element Msg
body { docs, selection } =
    column [ width fill, spacing 30 ]
        [ selectDoc docs
        , resultView docs selection
        ]


selectDoc : Dict.Dict String Document -> Element Msg
selectDoc dict =
    el
        [ width fill
        , Border.rounded 3
        , padding 30
        ]
        (Element.html (select [ onInput NewSelection ] (Dict.keys dict |> List.map strToOption)))


strToOption str =
    option [] [ Html.text str ]


resultView : Dict.Dict String Document -> Maybe String -> Element Msg
resultView docs selection =
    let
        get dict key =
            Dict.get key dict
    in
        case (selection |> Maybe.andThen (get docs)) of
            Just doc ->
                let
                    pos x y tag attrs =
                        tag ([ SAttr.x (String.fromFloat x), SAttr.y (String.fromFloat y) ] ++ attrs)

                    textStyle sz =
                        SAttr.style ("font-size: " ++ (String.fromInt sz) ++ "px;font-family: 'Source Code Pro', monospace;")

                    charWidth =
                        12

                    charHeight =
                        15

                    colorFromClass class =
                        case class of
                            "NAM-PER" ->
                                "#78CAD2"

                            "NAM-FAC" ->
                                "#63595C"

                            "NAM-LOC" ->
                                "#646881"

                            "NAM-ORG" ->
                                "#62BEC1"

                            "NAM-TTL" ->
                                "#5AD2F4"

                            "NAM-GPE" ->
                                "#72DDF7"

                            "NOM-PER" ->
                                "#F865B0"

                            "NOM-FAC" ->
                                "#E637BF"

                            "NOM-LOC" ->
                                "#FF928B"

                            "NOM-ORG" ->
                                "#FEC3A6"

                            "NOM-TTL" ->
                                "#FF3C38"

                            "NOM-GPE" ->
                                "#BB8588"

                            _ ->
                                "red"

                    mark string =
                        let
                            length =
                                String.length string

                            w =
                                length * charWidth + 8 |> String.fromInt

                            height =
                                charHeight + 8

                            h =
                                height |> String.fromFloat

                            padding =
                                4
                        in
                            Svg.svg [ SAttr.width w, SAttr.height h, SAttr.viewBox ("0 0 " ++ w ++ " " ++ h) ]
                                [ Svg.g []
                                    [ pos 0
                                        0
                                        Svg.rect
                                        [ SAttr.width w
                                        , SAttr.height h
                                        , SAttr.rx "5"
                                        , SAttr.ry "5"
                                        , SAttr.style ("fill:" ++ colorFromClass string ++ ";stroke:black;stroke-width:1;opacity:0.5")
                                        ]
                                        []
                                    , pos padding (height - padding) Svg.text_ [ textStyle 20 ] [ Svg.text string ]
                                    ]
                                ]

                    annotate : Int -> String -> List Entity -> List (Html.Html Msg)
                    annotate origin string ent =
                        let
                            annotation attrs marks begin end =
                                Html.span ([] ++ attrs)
                                    ([ String.slice begin end string |> Html.text ] ++ marks)

                            plain begin end =
                                Html.text (String.slice begin end string)

                            marked class begin end =
                                Html.div
                                    [ style "display" "inline-flex"
                                    , style "flex-direction" "column"
                                    , style "height" "3em"
                                    ]
                                    [ Html.div
                                        [ style "flex" "0 1 auto"
                                        , style "text-align" "center"
                                        , style "border" ("1px solid " ++ colorFromClass class)
                                        , style "border-radius" "5px"
                                        ]
                                        [ plain begin end ]
                                    , Html.div [ style "flex" "0 1 auto", style "text-align" "center" ] [ Html.div [] [ mark class ] ]
                                    ]
                        in
                            case ent of
                                { start, stop, class } :: tail ->
                                    (plain origin start) :: (marked class start stop) :: (annotate stop string tail)

                                [] ->
                                    []

                    label =
                        List.map
                            (\str ->
                                el []
                                    (Html.span [ colorFromClass str |> style "background-color" ] [ Html.text str ] |> Element.html)
                            )
                            [ "NAM-PER", "NAM-FAC", "NAM-LOC", "NAM-ORG", "NAM-TTL", "NAM-GPE", "NOM-PER", "NOM-FAC", "NOM-LOC", "NOM-ORG", "NOM-TTL", "NOM-GPE" ]
                in
                    column [ width fill ]
                        [ annotate 0 doc.text doc.entities |> Html.div [ style "font-family" "'Source Sans Pro', sans-serif" ] |> Element.html
                        ]

            Nothing ->
                none


errorString error =
    case error of
        Http.BadBody str ->
            "BadBody: " ++ str

        Http.BadStatus code ->
            "BadStatus: " ++ (String.fromInt code)

        Http.BadUrl str ->
            "BadUrl: " ++ str

        Http.NetworkError ->
            "NetworkError"

        Http.Timeout ->
            "Timeout"



-- HTTP


localApi : String
localApi =
    Url.absolute [ "gold" ] []


getDocuments : Cmd Msg
getDocuments =
    Http.get
        { url = localApi
        , expect = Http.expectJson NewDocuments (dict documentDecoder)
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
